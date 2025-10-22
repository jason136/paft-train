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
    ) -> None:
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "paft.xml")

        EzPickle.__init__(
            self,
            xml_path=xml_path,
            frame_skip=frame_skip,
            domain_randomize=domain_randomize,
            render_mode=render_mode,
        )

        self._domain_randomize = domain_randomize

        super().__init__(
            model_path=xml_path,
            frame_skip=frame_skip,
            observation_space=None,
            render_mode=render_mode,
        )

        # Define action space from number of actuators
        # Actions: [left_swing_angle, left_shrug_force, right_swing_angle, right_shrug_force]
        # Swing actuators (position servos): -π to π radians
        # Shrug actuators (linear motors): -10 to 10 N force
        action_dim = self.model.nu
        self.action_space = spaces.Box(
            low=np.array([-np.pi, -10.0, -np.pi, -10.0], dtype=np.float32),
            high=np.array([np.pi, 10.0, np.pi, 10.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: MuJoCo sensors(10) + relative_heading(2) = 12 total
        # Sensors: gyro(3) + accel(3) + left_shoulder(1) + right_shoulder(1) + left_arm(1) + right_arm(1)
        obs_dim = 10 + 2  # = 12 total
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._prev_x: Optional[float] = None
        self._target_heading: float = 0.0  # target angle in radians

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

    def _update_target_marker(self) -> None:
        """Update the simple target marker position - only called at reset."""
        robot_pos = self.data.qpos[0:2]  # x, y position
        
        # Place target marker 2m away in the desired direction
        target_distance = 2.0
        target_x = robot_pos[0] + target_distance * np.cos(self._target_heading)
        target_y = robot_pos[1] + target_distance * np.sin(self._target_heading)
        
        # Update target marker position
        marker_site_id = self.model.site("target_marker").id
        self.model.site_pos[marker_site_id] = [target_x, target_y, 0.1]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Store position before step
        x_before = float(self.data.qpos[0])
        y_before = float(self.data.qpos[1])

        self.do_simulation(action, self.frame_skip)

        # Get position after step
        x_after = float(self.data.qpos[0])
        y_after = float(self.data.qpos[1])

        # Calculate velocity in world coordinates
        dx = x_after - x_before
        dy = y_after - y_before
        velocity_vec = np.array([dx, dy])

        velocity_magnitude = np.linalg.norm(velocity_vec)

        # Reward for moving in target direction (dot product)
        target_dir = np.array(
            [np.cos(self._target_heading), np.sin(self._target_heading)]
        )
        velocity_toward_target = np.dot(velocity_vec, target_dir)

        # Control cost - encourage efficient movement (reduced to allow more exploration)
        ctrl_cost = 1e-3 * float(
            np.sum(np.square(action))
        )  # Very low penalty for trying things

        # Body angle penalty: discourage robot from laying down below 45 degrees
        # Calculate pitch angle from quaternion (body tilt forward/backward)
        qw, qx, qy, qz = (
            self.data.qpos[3],
            self.data.qpos[4],
            self.data.qpos[5],
            self.data.qpos[6],
        )

        # Convert quaternion to pitch angle (rotation around Y axis)
        # pitch = arcsin(2 * (qw * qy - qz * qx))
        pitch_angle = np.arcsin(2 * (qw * qy - qz * qx))
        pitch_degrees = np.abs(
            np.degrees(pitch_angle)
        )  # Get absolute angle from upright

        # Calculate body angle penalty - increasing penalty when further from upright
        if pitch_degrees > 45.0:  # Robot is tilted more than 45 degrees from upright
            # Exponentially increasing penalty as robot lays down more
            angle_penalty = (
                -0.1 * (pitch_degrees - 45.0) ** 2 / 45.0
            )  # Quadratic penalty
        else:
            angle_penalty = 0.0

        z_height = float(self.data.qpos[2])

        # Encourage leg movement (gait) - legs should be active, not static
        # Joint order: left_swing, left_shrug, right_swing, right_shrug
        left_swing_vel = abs(self.data.qvel[6])  # left swing velocity
        left_shrug_vel = abs(self.data.qvel[7])  # left shrug velocity
        right_swing_vel = abs(self.data.qvel[8])  # right swing velocity
        right_shrug_vel = abs(self.data.qvel[9])  # right shrug velocity
        leg_activity = (
            left_swing_vel + left_shrug_vel + right_swing_vel + right_shrug_vel
        )

        reward = (
            2.0 * velocity_toward_target  # Scaled down from 20
            + 1.0 * velocity_magnitude  # Scaled down from 5
            + 0.2 * leg_activity  # Scaled down from 2
            + angle_penalty  # New: penalty for laying down below 45 degrees
        )

        observation = self._get_obs()
        terminated = z_height < 0.05  # Episode ends if robot falls
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def reset_model(self) -> np.ndarray:
        if self._domain_randomize:
            self._randomize_domain()

        # Randomize target heading (0 to 2π radians)
        self._target_heading = self.np_random.uniform(0, 2 * np.pi)

        # Conservative initial randomization - prevent ground clipping while maintaining exploration
        qpos = self.init_qpos.copy()

        # Small position variations - keep robot well above ground
        qpos[0] += self.np_random.uniform(-0.05, 0.05)  # x position: ±5cm
        qpos[1] += self.np_random.uniform(-0.05, 0.05)  # y position: ±5cm
        qpos[2] += self.np_random.uniform(
            0.0, 0.03
        )  # z height: only UP 0-3cm (prevent ground clipping)

        # Small yaw variations for diverse starting orientations
        yaw = self.np_random.uniform(-0.3, 0.3)  # ±17 degrees
        qpos[3] = np.cos(yaw / 2)  # w
        qpos[4] = 0.0  # x
        qpos[5] = 0.0  # y
        qpos[6] = np.sin(yaw / 2)  # z

        # MUCH smaller joint angle variations to prevent clipping
        # Joint order: left_swing, left_shrug, right_swing, right_shrug

        # Swing joints: SAFE angular ranges to prevent ground penetration
        qpos[7] = self.np_random.uniform(
            -0.3, 0.3
        )  # left swing: ±17 degrees (vs ±69 before!)
        qpos[9] = self.np_random.uniform(-0.3, 0.3)  # right swing: ±17 degrees

        # Shrug joints: Small vertical variations
        qpos[8] = self.np_random.uniform(
            -0.01, 0.01
        )  # left shrug: ±1cm (vs ±3cm before)
        qpos[10] = self.np_random.uniform(-0.01, 0.01)  # right shrug: ±1cm

        # Realistic initial velocities - like a robot gently starting up
        qvel = self.init_qvel.copy()
        qvel[0] = self.np_random.uniform(-0.1, 0.1)  # gentle x velocity
        qvel[1] = self.np_random.uniform(-0.1, 0.1)  # gentle y velocity

        # Small rotational velocities - no wild spinning
        qvel[3] = self.np_random.uniform(-0.5, 0.5)  # gentle roll
        qvel[4] = self.np_random.uniform(-0.5, 0.5)  # gentle pitch
        qvel[5] = self.np_random.uniform(-0.5, 0.5)  # gentle yaw

        # Gentle joint movement - realistic servo startup
        qvel[6] = self.np_random.uniform(-0.5, 0.5)  # left swing: gentle start
        qvel[7] = self.np_random.uniform(-0.2, 0.2)  # left shrug: gentle start
        qvel[8] = self.np_random.uniform(-0.5, 0.5)  # right swing: gentle start
        qvel[9] = self.np_random.uniform(-0.2, 0.2)  # right shrug: gentle start

        self.set_state(qpos, qvel)
        self._prev_x = float(self.data.qpos[0])
        
        # Update target marker for debugging
        self._update_target_marker()
        
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
