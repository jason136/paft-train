import os
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils import EzPickle


class PaftMujocoEnv(MujocoEnv, EzPickle):
    """
    Minimal MuJoCo locomotion-style environment for the Mini TARS model.

    Observations: concatenated qpos (excluding absolute z and orientation to reduce trivialities)
    and qvel. Actions: position targets for hinge actuators in radians.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        xml_path: Optional[str] = None,
        frame_skip: int = 5,
        domain_randomize: bool = False,
        render_mode: Optional[str] = None,
        camera_name: str = "track",
        width: int = 720,
        height: int = 720,
    ) -> None:
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "paft.xml")

        EzPickle.__init__(
            self,
            xml_path=xml_path,
            frame_skip=frame_skip,
            domain_randomize=domain_randomize,
            render_mode=render_mode,
            camera_name=camera_name,
            width=width,
            height=height,
        )

        self._domain_randomize = domain_randomize

        super().__init__(
            model_path=xml_path,
            frame_skip=frame_skip,
            observation_space=None,
            render_mode=render_mode,
            camera_name=camera_name,
            width=width,
            height=height,
        )

        # Define action space from number of actuators
        # Actions: [left_swing_angle, left_shrug_force, right_swing_angle, right_shrug_force]
        # Network outputs normalized -1 to 1, we scale them appropriately:
        # - Swing: -1 to 1 → -π to π radians
        # - Shrug: -1 to 1 → -50 to 50 N force
        action_dim = self.model.nu
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

        # Action scaling factors (applied in step function)
        self._action_scale = np.array([np.pi, 50.0, np.pi, 50.0], dtype=np.float32)

        # Observation space: MuJoCo sensors(10) + relative_heading(2) = 12 total
        # Sensors: gyro(3) + accel(3) + left_shoulder(1) + right_shoulder(1) + left_arm(1) + right_arm(1)
        obs_dim = 10 + 2  # = 12 total
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._prev_x: Optional[float] = None
        self._target_heading: float = 0.0  # target angle in radians

        # Track previous joint velocities to measure smoothness
        self._prev_joint_vel: Optional[np.ndarray] = None
        # Track previous swing positions to detect crossing through center
        self._prev_swing_pos: Optional[np.ndarray] = None

    def _get_obs(self) -> np.ndarray:
        # Get sensor readings from MuJoCo sensors (first 10 values)
        # Order: gyro(3) + accel(3) + left_shoulder(1) + right_shoulder(1) + left_arm(1) + right_arm(1)
        sensor_data = self.data.sensordata[:10].copy().astype(np.float32)

        # Calculate relative heading: target direction in robot's local coordinate frame
        # Extract quaternion for coordinate transformation
        qw, qx, qy, qz = (
            self.data.qpos[3],
            self.data.qpos[4],
            self.data.qpos[5],
            self.data.qpos[6],
        )

        # Convert quaternion to rotation matrix (world to body transform)
        R = np.array(
            [
                [
                    1 - 2 * (qy**2 + qz**2),
                    2 * (qx * qy - qw * qz),
                    2 * (qx * qz + qw * qy),
                ],
                [
                    2 * (qx * qy + qw * qz),
                    1 - 2 * (qx**2 + qz**2),
                    2 * (qy * qz - qw * qx),
                ],
                [
                    2 * (qx * qz - qw * qy),
                    2 * (qy * qz + qw * qx),
                    1 - 2 * (qx**2 + qy**2),
                ],
            ],
            dtype=np.float32,
        )

        # Transform target direction to robot's local frame
        world_target_dir = np.array(
            [np.cos(self._target_heading), np.sin(self._target_heading), 0.0],
            dtype=np.float32,
        )
        local_target_dir_3d = R @ world_target_dir
        local_target_dir = local_target_dir_3d[0:2]  # Take only x,y components

        # Concatenate sensor data with relative heading
        obs = np.concatenate(
            [
                sensor_data,  # 10 values: all MuJoCo sensor readings
                local_target_dir,  # 2 values: relative heading in local frame
            ]
        ).astype(np.float32)

        return obs

    def _update_target_arrow(self) -> None:
        x, y, z = self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]
        hover_height = 0.25

        geom_id = self.model.geom("target_arrow").id
        arrow_half_length = self.model.geom_size[geom_id][0]

        arrow_center_x = x + arrow_half_length * np.cos(self._target_heading)
        arrow_center_y = y + arrow_half_length * np.sin(self._target_heading)

        self.model.geom_pos[geom_id] = [
            arrow_center_x,
            arrow_center_y,
            z + hover_height,
        ]

        cos_h = np.cos(self._target_heading)
        sin_h = np.sin(self._target_heading)

        # Rotation matrix for Z-axis rotation (yaw)
        rotation_matrix = np.array(
            [[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]], dtype=np.float64
        ).flatten()

        self.data.geom_xmat[geom_id] = rotation_matrix

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Clip normalized action to [-1, 1]
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Scale action from [-1, 1] to actual actuator ranges
        scaled_action = action * self._action_scale

        # Store position and joint velocities before step
        x_before = float(self.data.qpos[0])
        y_before = float(self.data.qpos[1])

        self.do_simulation(scaled_action, self.frame_skip)

        # Get position after step
        x_after = float(self.data.qpos[0])
        y_after = float(self.data.qpos[1])

        # Calculate velocity in world coordinates
        dx = x_after - x_before
        dy = y_after - y_before
        velocity_vec = np.array([dx, dy])

        # Calculate velocity magnitude
        velocity_magnitude = np.linalg.norm(velocity_vec)

        # Reward for moving in target direction (dot product)
        target_dir = np.array(
            [np.cos(self._target_heading), np.sin(self._target_heading)]
        )
        velocity_toward_target = np.dot(velocity_vec, target_dir)

        qw, qx, qy, qz = (
            self.data.qpos[3],
            self.data.qpos[4],
            self.data.qpos[5],
            self.data.qpos[6],
        )

        # Heading alignment: reward robot for facing the target direction
        # Extract robot's current heading (yaw) from quaternion
        robot_yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        # Calculate heading error (difference between robot's heading and target)
        heading_error = np.abs(
            np.arctan2(
                np.sin(robot_yaw - self._target_heading),
                np.cos(robot_yaw - self._target_heading),
            )
        )
        heading_alignment = np.cos(heading_error)

        # Joint activity rewards: encourage LARGE, SMOOTH sweeping motions
        # Penalize rapid direction changes (jittering)

        # Current joint positions and velocities
        joint_pos = self.data.qpos[
            7:11
        ]  # left_swing, left_shrug, right_swing, right_shrug
        joint_vel = self.data.qvel[
            6:10
        ]  # left_swing, left_shrug, right_swing, right_shrug

        # 1. HIGH VELOCITY: Reward fast joint movements (linear to be more stable)
        swing_velocity = (abs(joint_vel[0]) + abs(joint_vel[2])) / 2.0
        shrug_velocity = (abs(joint_vel[1]) + abs(joint_vel[3])) / 2.0
        # Linear velocity reward is more stable than squared
        velocity_reward = swing_velocity + shrug_velocity

        # 2. SMOOTHNESS: Penalize rapid changes in velocity (jerk)
        # Jittering = high acceleration changes, smooth swings = consistent velocity
        if self._prev_joint_vel is not None:
            # Measure how much velocity changed (acceleration/jerk)
            vel_change = np.abs(joint_vel - self._prev_joint_vel)
            swing_jerk = (vel_change[0] + vel_change[2]) / 2.0
            shrug_jerk = (vel_change[1] + vel_change[3]) / 2.0
            total_jerk = swing_jerk + shrug_jerk
            # Penalize jerkiness (subtract from reward)
            smoothness_penalty = (
                total_jerk * 1.0
            )  # Reduced penalty to allow more movement
        else:
            smoothness_penalty = 0.0

        # Store current velocities for next step
        self._prev_joint_vel = joint_vel.copy()

        # 3. RANGE OF MOTION: Reward using extreme positions
        swing_range = (abs(joint_pos[0]) + abs(joint_pos[2])) / 2.0
        shrug_range = (abs(joint_pos[1]) + abs(joint_pos[3])) / 2.0
        range_reward = swing_range + shrug_range

        # 4. CROSSING REWARD: Reward when arms swing through the body's neutral plane
        # Detect sign changes (crossing zero) for swing joints
        crossing_reward = 0.0
        if self._prev_swing_pos is not None:
            left_swing_pos = joint_pos[0]
            right_swing_pos = joint_pos[2]
            prev_left = self._prev_swing_pos[0]
            prev_right = self._prev_swing_pos[1]

            # Check if left arm crossed through zero (changed sign)
            if (prev_left * left_swing_pos) < 0:  # Sign change means crossing
                crossing_reward += 1.0

            # Check if right arm crossed through zero
            if (prev_right * right_swing_pos) < 0:
                crossing_reward += 1.0

        # Store current swing positions for next step
        self._prev_swing_pos = np.array([joint_pos[0], joint_pos[2]])

        # 5. ACTION MAGNITUDE: Reward bold, decisive commands
        action_magnitude = np.linalg.norm(action) / np.sqrt(len(action))
        bold_action_reward = action_magnitude**2

        # 6. SYNC REWARD: Reward moving arms in sync
        # Product of velocities: positive when moving in same direction
        sync_reward = joint_vel[0] * joint_vel[2] + joint_vel[1] * joint_vel[3]

        # Combine: reward fast, smooth, large-range movements with crossing
        activity_reward = (
            0.25 * velocity_reward  # Fast movements
            - 1.0 * smoothness_penalty  # Moderate penalty for jerkiness
            # + 1.0 * range_reward  # Use full range of motion
            + 10.0
            * crossing_reward  # Strong reward for crossing neutral plane (cycles)
            + 1.0 * sync_reward  # Small reward for moving arms in sync
            # + 0.1 * bold_action_reward  # Bold actions
        )

        # EXPLORATION-FRIENDLY REWARD: Balanced task + exploration
        reward = (
            10.0 * max(0, velocity_toward_target)  # PRIMARY: move toward target!
            # + 5.0 * velocity_magnitude  # SECONDARY: any movement is good
            + 5.0 * max(0, heading_alignment)  # TERTIARY: face target direction
            + 1.0 * activity_reward  # QUATERNARY: activity reward
        )

        observation = self._get_obs()

        # Calculate tilt angle from quaternion (angle from vertical z-axis)
        # For a unit quaternion, the z-component of the up vector is: 1 - 2*(qx^2 + qy^2)
        up_z = 1 - 2 * (qx**2 + qy**2)
        up_z = np.clip(up_z, -1.0, 1.0)  # Numerical safety
        tilt_angle = np.arccos(up_z)  # Angle from vertical in radians

        # Terminate if body tilts beyond 75° from vertical (15° from horizontal)
        terminated = bool(tilt_angle > (np.pi * 75 / 180))  # 75 degrees = 1.309 radians

        truncated = False
        info = {}

        self._update_target_arrow()

        return observation, reward, terminated, truncated, info

    def reset_model(self) -> np.ndarray:
        if self._domain_randomize:
            self._randomize_domain()

        # Randomize target heading (0 to 2π radians)
        self._target_heading = self.np_random.uniform(0, 2 * np.pi)

        # STABLE initial states for learning locomotion from scratch
        qpos = self.init_qpos.copy()

        # Small position variations to prevent overfitting to spawn location
        qpos[0] += self.np_random.uniform(-0.1, 0.1)  # x position: ±10cm
        qpos[1] += self.np_random.uniform(-0.1, 0.1)  # y position: ±10cm
        qpos[2] += self.np_random.uniform(0.0, 0.005)  # z height: 0-0.5cm up (grounded)

        # Varied yaw for diverse starting orientations
        yaw = self.np_random.uniform(-np.pi, np.pi)  # Full 360 degrees is fine
        qpos[3] = np.cos(yaw / 2)  # w
        qpos[4] = 0.0  # x
        qpos[5] = 0.0  # y
        qpos[6] = np.sin(yaw / 2)  # z

        # Reasonable joint positions - within normal operating range
        # Joint order: left_swing, left_shrug, right_swing, right_shrug

        # Swing joints: Start with arms in front of body (approx 30-85 degrees)
        qpos[7] = self.np_random.uniform(0.5, 1.5)  # left swing: ~30-85 degrees
        qpos[9] = self.np_random.uniform(0.5, 1.5)  # right swing: ~30-85 degrees

        # Shrug joints: small variation around neutral
        qpos[8] = self.np_random.uniform(-0.01, 0.01)  # left shrug: ±1cm
        qpos[10] = self.np_random.uniform(-0.01, 0.01)  # right shrug: ±1cm

        # Start from REST - let the agent learn to generate movement
        qvel = self.init_qvel.copy()
        # All velocities remain at zero (init_qvel should be zeros)

        self.set_state(qpos, qvel)
        self._prev_x = float(self.data.qpos[0])
        self._prev_joint_vel = None  # Reset joint velocity tracking
        self._prev_swing_pos = None  # Reset swing position tracking

        self._update_target_arrow()

        return self._get_obs()

    def _randomize_domain(self) -> None:
        mass_scale = self.np_random.uniform(0.8, 1.2)
        self.model.body_mass[:] *= mass_scale
        friction_scale = self.np_random.uniform(0.8, 1.2)
        self.model.geom_friction[:, 0] *= friction_scale  # sliding friction


def _make_env(**kwargs) -> gym.Env:
    return PaftMujocoEnv(**kwargs)


def register_env() -> None:
    gym.register(
        id="Paft-v0",
        entry_point="paft_env:PaftMujocoEnv",
        max_episode_steps=10000,
    )


try:
    register_env()
except Exception:
    pass
