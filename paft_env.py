import os
from typing import Optional, Tuple, Dict
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils import EzPickle


class PaftEnv(MujocoEnv, EzPickle):
    """
    Locomotion environment for PAFT robot.

    No joint encoders — the policy infers joint state from IMU + action history.
    A learnable gait clock lets the policy choose its own stride frequency.

    Observation layout:
        IMU gyro (3) + IMU accel (3) + relative heading (2)
        + gait phase sin/cos (2) + action history (10×4=40) = 50 dims
    Action: normalized [-1, 1] for 4 actuators + 1 gait frequency = 5 dims
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    NUM_IMU = 6           # gyro(3) + accel(3)
    NUM_HEADING = 2       # target direction in local frame
    NUM_PHASE = 2         # sin(2πφ), cos(2πφ)
    NUM_MOTORS = 4        # left_swing, left_shrug, right_swing, right_shrug
    ACTION_HISTORY_LEN = 10

    # Gait frequency bounds (Hz) — policy chooses within this range
    FREQ_MIN = 0.5   # slowest gait: 2.0s period
    FREQ_MAX = 3.0   # fastest gait: 0.33s period

    def __init__(
        self,
        xml_path: Optional[str] = None,
        frame_skip: int = 10,
        render_mode: Optional[str] = None,
        width: int = 480,
        height: int = 480,
    ) -> None:
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "paft.xml")

        EzPickle.__init__(self, xml_path, frame_skip, render_mode, width, height)

        super().__init__(
            model_path=xml_path,
            frame_skip=frame_skip,
            observation_space=None,
            render_mode=render_mode,
            camera_name="track",
            width=width,
            height=height,
        )

        # Actions: 4 motor commands + 1 gait frequency, all normalized [-1, 1]
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.NUM_MOTORS + 1,), dtype=np.float32
        )
        self._action_scale = np.array([np.pi, 50.0, np.pi, 50.0], dtype=np.float32)

        # Action history ring buffer — motor actions only (oldest first)
        self._action_history = np.zeros(
            (self.ACTION_HISTORY_LEN, self.NUM_MOTORS), dtype=np.float32
        )

        # Gait phase clock
        self._gait_phase = 0.0  # [0, 1) — wraps every cycle

        # Observations: IMU(6) + heading(2) + phase(2) + action_history(10×4)
        obs_dim = (
            self.NUM_IMU
            + self.NUM_HEADING
            + self.NUM_PHASE
            + self.ACTION_HISTORY_LEN * self.NUM_MOTORS
        )
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._target_heading = 0.0
        self._step_count = 0

    def _get_obs(self) -> np.ndarray:
        """IMU + relative heading + gait phase clock + flattened action history."""
        # IMU only: gyro(3) + accelerometer(3) = 6
        imu = self.data.sensordata[: self.NUM_IMU].astype(np.float32)

        # Robot orientation quaternion → rotation matrix (world → body)
        qw, qx, qy, qz = self.data.qpos[3:7]
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

        # Target direction in local frame
        world_dir = np.array(
            [np.cos(self._target_heading), np.sin(self._target_heading), 0.0]
        )
        local_dir = (R @ world_dir)[:2]

        # Gait phase clock
        phase_signal = np.array(
            [
                np.sin(2 * np.pi * self._gait_phase),
                np.cos(2 * np.pi * self._gait_phase),
            ],
            dtype=np.float32,
        )

        # Flatten motor action history (oldest → newest)
        history_flat = self._action_history.flatten()

        return np.concatenate([imu, local_dir, phase_signal, history_flat]).astype(
            np.float32
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0)

        # Split: 4 motor commands + 1 gait frequency
        motor_action = action[: self.NUM_MOTORS]
        freq_action = action[self.NUM_MOTORS]  # in [-1, 1]

        # Map [-1, 1] → [FREQ_MIN, FREQ_MAX] Hz
        gait_freq = self.FREQ_MIN + (freq_action + 1.0) / 2.0 * (
            self.FREQ_MAX - self.FREQ_MIN
        )

        # Advance gait phase clock
        dt = self.frame_skip * self.model.opt.timestep
        self._gait_phase = (self._gait_phase + gait_freq * dt) % 1.0

        # Push motor action into history (FIFO: drop oldest, append newest)
        self._action_history = np.roll(self._action_history, shift=-1, axis=0)
        self._action_history[-1] = motor_action

        x_before, y_before = self.data.qpos[0], self.data.qpos[1]

        self.do_simulation(motor_action * self._action_scale, self.frame_skip)
        self._step_count += 1

        x_after, y_after = self.data.qpos[0], self.data.qpos[1]
        qw, qx, qy, qz = self.data.qpos[3:7]

        # === REWARD ===
        # 1. Forward velocity toward target
        velocity = np.array([x_after - x_before, y_after - y_before])
        target_dir = np.array(
            [np.cos(self._target_heading), np.sin(self._target_heading)]
        )
        forward_vel = np.dot(velocity, target_dir)

        # 2. Heading alignment (face the target direction)
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        heading_error = np.abs(
            np.arctan2(
                np.sin(yaw - self._target_heading), np.cos(yaw - self._target_heading)
            )
        )
        alignment = np.cos(heading_error)  # 1 when aligned, -1 when opposite

        # 3. Cost of time — standing still bleeds reward, so the only way
        #    to stay positive is sustained forward motion. Forces the policy
        #    to explore recovery/repositioning rather than lunge-then-idle.
        time_cost = 0.1

        reward = (
            forward_vel * 100.0   # Velocity toward target
            + alignment * 0.1     # Small bonus for facing target
            - time_cost           # Must keep moving to stay positive
        )

        # === TERMINATION ===
        up_z = 1 - 2 * (qx**2 + qy**2)
        tilt_angle = np.arccos(np.clip(up_z, -1, 1))
        terminated = bool(tilt_angle > np.radians(60))  # 60° tilt = fall

        # Penalty for falling — prevents "fall early to stop the bleeding"
        if terminated:
            reward -= 10.0

        self._update_arrow()

        return self._get_obs(), reward, terminated, False, {"forward_vel": forward_vel}

    def reset_model(self) -> np.ndarray:
        self._step_count = 0
        self._action_history[:] = 0.0
        self._gait_phase = 0.0

        # Light domain randomization
        if hasattr(self, "_base_masses"):
            self.model.body_mass[:] = self._base_masses * self.np_random.uniform(
                0.95, 1.05
            )
        else:
            self._base_masses = self.model.body_mass.copy()

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Randomize target heading
        self._target_heading = self.np_random.uniform(0, 2 * np.pi)

        # Small position variation
        qpos[0] += self.np_random.uniform(-0.05, 0.05)
        qpos[1] += self.np_random.uniform(-0.05, 0.05)

        # Start facing the target (± small noise) so the robot can learn
        # forward locomotion first; turning emerges from heading commands
        # Robot forward is +Y, but heading 0 is +X, so subtract π/2
        yaw = self._target_heading - np.pi / 2 + self.np_random.uniform(-0.3, 0.3)
        qpos[3], qpos[6] = np.cos(yaw / 2), np.sin(yaw / 2)

        # Arms in consistent neutral position
        qpos[7] = qpos[9] = 0.0

        self.set_state(qpos, qvel)
        self._update_arrow()
        return self._get_obs()

    def _update_arrow(self) -> None:
        """Update visual target arrow."""
        x, y, z = self.data.qpos[:3]
        geom_id = self.model.geom("target_arrow").id
        arrow_len = self.model.geom_size[geom_id][0]

        self.model.geom_pos[geom_id] = [
            x + arrow_len * np.cos(self._target_heading),
            y + arrow_len * np.sin(self._target_heading),
            z + 0.25,
        ]

        cos_h, sin_h = np.cos(self._target_heading), np.sin(self._target_heading)
        self.data.geom_xmat[geom_id] = np.array(
            [[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]]
        ).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────


def register_env() -> None:
    gym.register(id="Paft-v0", entry_point="paft_env:PaftEnv", max_episode_steps=1000)


try:
    register_env()
except Exception:
    pass
