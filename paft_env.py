import os
from typing import Optional, Tuple, Dict
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils import EzPickle


class PaftEnv(MujocoEnv, EzPickle):
    """
    Locomotion environment for PAFT robot (teacher policy).

    Observation: IMU(6) + heading(2) + phase(2) + joint_pos(4) + joint_vel(4) = 18 dims
    Action: normalized [-1, 1] → swing ±1.0 rad, shrug ±20mm
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    NUM_IMU = 6  # gyro(3) + accel(3)
    NUM_HEADING = 2  # target direction in local frame
    NUM_PHASE = 2  # sin(2πφ), cos(2πφ)
    NUM_MOTORS = 4  # left_swing, left_shrug, right_swing, right_shrug
    NUM_JOINT_POS = 4  # actual joint positions (swing×2 + shrug×2)
    NUM_JOINT_VEL = 4  # actual joint velocities

    GAIT_FREQ = 0.15  # Hz — 6.7s cycle, slow enough for gravity on both halves
    SWING_AMPLITUDE = 0.70  # radians (~40°) — arms reach far out to tip body
    SHRUG_AMPLITUDE = 0.020  # meters (20mm) — full actuator range
    PITCH_AMPLITUDE = 0.10  # radians (~5.7°) — body rock follows arm reach

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

        # Actions: 4 motor commands, normalized [-1, 1]
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.NUM_MOTORS,), dtype=np.float32
        )
        self._action_scale = np.array([1.0, 0.02, 1.0, 0.02], dtype=np.float32)

        self._gait_phase = 0.0  # [0, 1) — wraps every cycle
        self._last_action = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self._prev_action = np.zeros(self.NUM_MOTORS, dtype=np.float32)

        obs_dim = (
            self.NUM_IMU
            + self.NUM_HEADING
            + self.NUM_PHASE
            + self.NUM_JOINT_POS
            + self.NUM_JOINT_VEL
        )
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._target_heading = 0.0
        self._step_count = 0

    def _gait_targets(self, phase: float) -> Tuple[float, float, float]:
        """Piecewise gait waveform: lift → reach → tilt → pull → push → recover.

        Returns (swing_target, shrug_target, pitch_target).

        Shoulders UP lifts arms off ground. Arms swing forward in air.
        Gravity tilts body forward, arms contact ground. Shoulders
        equalize while body pulls forward. Shoulders push DOWN to
        lift body off floor. Body swings forward and lands. Recover.

        Cycle phases (fraction of period):
          0.00–0.05  LIFT: shrug 0 → +S                  (shoulders up, clear ground)
          0.05–0.15  REACH: swing 0 → +A, shrug +S       (arms forward in air)
          0.15–0.33  HOLD: swing +A, shrug +S              (wait for gravity tilt)
          0.33–0.71  PULL: swing +A → 0, shrug +S → -S    (shoulders settle past
                                                           neutral as body follows)
          0.71–0.77  PUSH: swing 0 → -A, shrug -S         (arms back, body lifts)
          0.77–0.87  HOLD2: hold -A, shrug -S              (wait for body swing fwd)
          0.87–0.95  RECOVER: swing -A → 0, shrug -S → 0  (arms fwd, equalize)
          0.95–1.00  SETTLE: swing 0, shrug 0              (prep next cycle)
        """
        A = self.SWING_AMPLITUDE
        S = self.SHRUG_AMPLITUDE

        if phase < 0.05:
            frac = phase / 0.05
            swing = 0.0
            shrug = S * frac
        elif phase < 0.15:
            frac = (phase - 0.05) / 0.10
            swing = A * frac
            shrug = S
        elif phase < 0.33:
            swing = A
            shrug = S
        elif phase < 0.71:
            frac = (phase - 0.33) / 0.38
            swing = A * (1.0 - frac)
            shrug = S * (1.0 - 2.0 * frac)
        elif phase < 0.77:
            frac = (phase - 0.71) / 0.06
            swing = -A * frac
            shrug = -S
        elif phase < 0.87:
            swing = -A
            shrug = -S
        elif phase < 0.95:
            frac = (phase - 0.87) / 0.08
            swing = -A * (1.0 - frac)
            shrug = -S * (1.0 - frac)
        else:
            swing = 0.0
            shrug = 0.0

        pitch = self.PITCH_AMPLITUDE * (swing / A) if A > 0 else 0.0
        return swing, shrug, pitch

    def _get_obs(self) -> np.ndarray:
        """IMU + heading + phase + joint state + action history."""
        imu = self.data.sensordata[: self.NUM_IMU].astype(np.float32)

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

        world_dir = np.array(
            [np.cos(self._target_heading), np.sin(self._target_heading), 0.0]
        )
        local_dir = (R @ world_dir)[:2]

        phase_signal = np.array(
            [
                np.sin(2 * np.pi * self._gait_phase),
                np.cos(2 * np.pi * self._gait_phase),
            ],
            dtype=np.float32,
        )

        joint_pos = self.data.qpos[7:11].astype(np.float32)
        joint_vel = self.data.qvel[6:10].astype(np.float32)

        return np.concatenate(
            [imu, local_dir, phase_signal, joint_pos, joint_vel]
        ).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0)

        # Advance gait phase clock at fixed frequency
        dt = self.frame_skip * self.model.opt.timestep
        self._gait_phase = (self._gait_phase + self.GAIT_FREQ * dt) % 1.0

        self._prev_action = self._last_action.copy()
        self._last_action = action.copy()

        x_before, y_before = self.data.qpos[0], self.data.qpos[1]

        self.do_simulation(action * self._action_scale, self.frame_skip)
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

        # 3. Gait tracking — loose guidance on timing, not exact trajectories.
        #    Low k values let the policy deviate from the waveform while
        #    still rewarding being in the right part of the cycle.
        swing_target, shrug_target, _pitch_target = self._gait_targets(
            self._gait_phase
        )

        left_swing = self.data.qpos[7]
        left_shrug = self.data.qpos[8]
        right_swing = self.data.qpos[9]
        right_shrug = self.data.qpos[10]

        # exp(-k * err²): k=5 means error of ~0.45 rad still gets 37% credit
        k_swing = 5.0
        k_shrug = 5.0 / (self.SHRUG_AMPLITUDE ** 2)  # normalized: full-range error → 37%

        gait_reward = (
            np.exp(-k_swing * (left_swing - swing_target) ** 2)
            + np.exp(-k_swing * (right_swing - swing_target) ** 2)
            + np.exp(-k_shrug * (left_shrug - shrug_target) ** 2)
            + np.exp(-k_shrug * (right_shrug - shrug_target) ** 2)
        )

        action_rate = np.sum((action - self._prev_action) ** 2)

        # Multiplicative gait: amplifies velocity reward when roughly
        # following the phase. 4 components → bonus range [1.0, 3.0]
        gait_bonus = 1.0 + gait_reward * 0.5

        reward = (
            forward_vel * 500.0 * gait_bonus  # velocity × gait quality
            + alignment * 0.05  # face the target
            - action_rate * 0.05  # smooth control
        )

        # === TERMINATION ===
        up_z = 1 - 2 * (qx**2 + qy**2)
        tilt_angle = np.arccos(np.clip(up_z, -1, 1))
        terminated = bool(tilt_angle > np.radians(60))  # 60° tilt = fall

        # Penalty for falling — prevents "fall early to stop the bleeding"
        if terminated:
            reward -= 10.0

        self._update_arrow()

        info = {
            "forward_vel": forward_vel,
            "gait_reward": gait_reward,
            "swing_error": 0.5
            * (abs(left_swing - swing_target) + abs(right_swing - swing_target)),
            "shrug_error": 0.5
            * (abs(left_shrug - shrug_target) + abs(right_shrug - shrug_target)),
            "gait_phase": self._gait_phase,
        }
        return self._get_obs(), reward, terminated, False, info

    def reset_model(self) -> np.ndarray:
        self._step_count = 0
        self._last_action = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self._prev_action = np.zeros(self.NUM_MOTORS, dtype=np.float32)
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

        # Fixed heading for Phase 1 — learn to walk forward first,
        # add heading randomization later once walking is reliable
        self._target_heading = 0.0

        qpos[0] += self.np_random.uniform(-0.05, 0.05)
        qpos[1] += self.np_random.uniform(-0.05, 0.05)

        # Face the target: robot forward is +Y, heading 0 is +X, so yaw = -π/2
        yaw = -np.pi / 2 + self.np_random.uniform(-0.1, 0.1)
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
