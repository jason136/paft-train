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

    SWING_AMPLITUDE = 0.70  # radians (~40°)
    SHRUG_AMPLITUDE = 0.005  # meters (5mm) — JQDML 10mm stroke centered

    # 4-phase self-paced gait: two symmetric halves (arms, then body)
    SHRUG_UP_THRESH = 0.00375  # m — 75% of shrug amplitude
    REACH_THRESH = 0.525  # rad — 75% of swing amplitude
    SHRUG_DOWN_THRESH = -0.00375  # m — 75% of shrug amplitude
    BODY_FWD_THRESH = -0.525  # rad — symmetric with REACH_THRESH
    NEUTRAL_SW_THRESH = 0.12  # rad — swing "neutral"
    NEUTRAL_SH_THRESH = 0.002  # m — shrug "neutral"

    PHASE_REWARD = 5.0  # per phase transition
    CYCLE_REWARD = 25.0  # full cycle completion bonus

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
        self._action_scale = np.array([1.5708, 0.005, 1.5708, 0.005], dtype=np.float32)

        self._gait_stage = 0  # 0=IDLE, 1=REACHED, 2=PULLED
        self._steps_completed = 0
        self._peak_swing = 0.0
        self._peak_shrug = 0.0
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

    def _check_gait_progress(self, avg_swing: float, avg_shrug: float) -> float:
        """4-phase self-paced gait with normalized shaping.

        Shaping scales are set so reaching full amplitude on either axis
        produces exactly PHASE_REWARD of shaping reward. No multipliers.

        0 FORWARD:  shrug up + swing forward
        1 SETTLE:   return to neutral
        2 BACKWARD: shrug down + swing back
        3 RECOVER:  return to neutral (cycle complete)
        """
        reward = 0.0
        sw_scale = self.PHASE_REWARD / self.SWING_AMPLITUDE
        sh_scale = self.PHASE_REWARD / self.SHRUG_AMPLITUDE
        settle_scale = self.PHASE_REWARD / (self.SWING_AMPLITUDE + 1.0)

        if self._gait_stage == 0:  # FORWARD
            new_sh = max(self._peak_shrug, avg_shrug)
            new_sw = max(self._peak_swing, avg_swing)
            reward += (new_sh - self._peak_shrug) * sh_scale
            reward += (new_sw - self._peak_swing) * sw_scale
            self._peak_shrug = new_sh
            self._peak_swing = new_sw
            reached = avg_swing > self.REACH_THRESH
            lifted = avg_shrug > self.SHRUG_UP_THRESH
            if reached or lifted:
                bonus = 1.0 + (1.0 if reached else 0.0) + (1.0 if lifted else 0.0)
                reward += self.PHASE_REWARD * bonus
                self._gait_stage = 1
                self._peak_swing = 0.0
                self._peak_shrug = 0.0

        elif self._gait_stage == 1:  # SETTLE
            curr_dist = abs(avg_swing) + abs(avg_shrug) * (
                self.SWING_AMPLITUDE / self.SHRUG_AMPLITUDE
            )
            if self._peak_swing == 0.0:
                self._peak_swing = curr_dist
            if curr_dist < self._peak_swing:
                reward += (self._peak_swing - curr_dist) * settle_scale
                self._peak_swing = curr_dist
            if abs(avg_swing) < self.NEUTRAL_SW_THRESH and abs(avg_shrug) < self.NEUTRAL_SH_THRESH:
                reward += self.PHASE_REWARD
                self._gait_stage = 2
                self._peak_swing = 0.0
                self._peak_shrug = 0.0

        elif self._gait_stage == 2:  # BACKWARD
            new_sh = min(self._peak_shrug, avg_shrug)
            new_sw = min(self._peak_swing, avg_swing)
            reward += (self._peak_shrug - new_sh) * sh_scale
            reward += (self._peak_swing - new_sw) * sw_scale
            self._peak_shrug = new_sh
            self._peak_swing = new_sw
            body_fwd = avg_swing < self.BODY_FWD_THRESH
            pushed = avg_shrug < self.SHRUG_DOWN_THRESH
            if body_fwd or pushed:
                bonus = 1.0 + (1.0 if body_fwd else 0.0) + (1.0 if pushed else 0.0)
                reward += self.PHASE_REWARD * bonus
                self._gait_stage = 3
                self._peak_swing = 0.0
                self._peak_shrug = 0.0

        elif self._gait_stage == 3:  # RECOVER
            curr_dist = abs(avg_swing) + abs(avg_shrug) * (
                self.SWING_AMPLITUDE / self.SHRUG_AMPLITUDE
            )
            if self._peak_swing == 0.0:
                self._peak_swing = curr_dist
            if curr_dist < self._peak_swing:
                reward += (self._peak_swing - curr_dist) * settle_scale
                self._peak_swing = curr_dist
            if abs(avg_swing) < self.NEUTRAL_SW_THRESH and abs(avg_shrug) < self.NEUTRAL_SH_THRESH:
                self._gait_stage = 0
                self._steps_completed += 1
                self._peak_swing = 0.0
                self._peak_shrug = 0.0
                reward += self.CYCLE_REWARD

        return reward

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

        stage_phase = self._gait_stage / 4.0
        phase_signal = np.array(
            [
                np.sin(2 * np.pi * stage_phase),
                np.cos(2 * np.pi * stage_phase),
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

        self._prev_action = self._last_action.copy()
        self._last_action = action.copy()

        x_before, y_before = self.data.qpos[0], self.data.qpos[1]

        self.do_simulation(action * self._action_scale, self.frame_skip)
        self._step_count += 1

        x_after, y_after = self.data.qpos[0], self.data.qpos[1]
        qw, qx, qy, qz = self.data.qpos[3:7]

        # === REWARD ===
        velocity = np.array([x_after - x_before, y_after - y_before])
        target_dir = np.array(
            [np.cos(self._target_heading), np.sin(self._target_heading)]
        )
        forward_vel = np.dot(velocity, target_dir)

        r_velocity = forward_vel * 200.0

        avg_swing = 0.5 * (self.data.qpos[7] + self.data.qpos[9])
        avg_shrug = 0.5 * (self.data.qpos[8] + self.data.qpos[10])
        r_gait = self._check_gait_progress(avg_swing, avg_shrug)

        r_survive = -0.05

        reward = r_velocity + r_gait + r_survive

        # === TERMINATION ===
        up_z = 1 - 2 * (qx**2 + qy**2)
        tilt_angle = np.arccos(np.clip(up_z, -1, 1))
        terminated = bool(tilt_angle > np.radians(60))

        if terminated:
            reward -= 10.0

        self._update_arrow()

        info = {
            "r_velocity": r_velocity,
            "r_gait": r_gait,
            "r_survive": r_survive,
            "gait_stage": self._gait_stage,
            "steps_completed": self._steps_completed,
        }
        return self._get_obs(), reward, terminated, False, info

    def set_start_stage(self, stage: int, deterministic: bool = False) -> None:
        """Force a specific starting stage for the next reset."""
        self._forced_stage = stage
        self._deterministic_reset = deterministic

    def reset_model(self) -> np.ndarray:
        self._step_count = 0
        self._last_action = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self._prev_action = np.zeros(self.NUM_MOTORS, dtype=np.float32)
        self._steps_completed = 0
        self._peak_swing = 0.0
        self._peak_shrug = 0.0

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        self._target_heading = 0.0

        deterministic = getattr(self, "_deterministic_reset", False)
        if deterministic:
            self._deterministic_reset = False
        else:
            qpos[0] += self.np_random.uniform(-0.05, 0.05)
            qpos[1] += self.np_random.uniform(-0.05, 0.05)

        if hasattr(self, "_forced_stage") and self._forced_stage is not None:
            stage = self._forced_stage
            self._forced_stage = None
        else:
            stage = int(self.np_random.integers(0, 4))
        self._gait_stage = stage

        yaw = -np.pi / 2
        if not deterministic:
            yaw += self.np_random.uniform(-0.1, 0.1)

        if stage == 0:
            qpos[7] = qpos[9] = 0.0
            qpos[8] = qpos[10] = 0.0
        elif stage == 1:
            # Arms in air: randomize swing and shrug (the airborne part)
            sw_noise = self.np_random.normal(0, self.SWING_AMPLITUDE * 0.10, size=2)
            sh_noise = self.np_random.normal(0, self.SHRUG_AMPLITUDE * 0.15, size=2)
            qpos[7] = self.SWING_AMPLITUDE + sw_noise[0]
            qpos[9] = self.SWING_AMPLITUDE + sw_noise[1]
            # Shrug must stay positive (arms in air, not clipping ground)
            qpos[8] = max(0.002, self.SHRUG_AMPLITUDE + sh_noise[0])
            qpos[10] = max(0.002, self.SHRUG_AMPLITUDE + sh_noise[1])
        elif stage == 2:
            qpos[7] = qpos[9] = 0.0
            qpos[8] = qpos[10] = 0.0
        elif stage == 3:
            # Body tilted forward, arms vertical and planted, resting on arms.
            # Compute body z so arm tips sit exactly at ground level:
            #   pivot_z = body_z + 0.1185 * cos(pitch)
            #   arm_bottom_z = pivot_z + shrug - 0.3585 = 0
            #   body_z = 0.3585 - shrug - 0.1185 * cos(pitch)
            pitch_noise = self.np_random.normal(0, self.SWING_AMPLITUDE * 0.10)
            pitch = max(0.15, self.SWING_AMPLITUDE + pitch_noise)
            shrug_val = -self.SHRUG_AMPLITUDE
            qpos[7] = qpos[9] = -pitch  # counter-rotate arms to stay vertical
            qpos[8] = qpos[10] = shrug_val
            cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
            cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
            qpos[3] = cy * cp
            qpos[4] = cy * sp
            qpos[5] = sy * sp
            qpos[6] = sy * cp
            qpos[2] = 0.3585 - shrug_val - 0.1185 * np.cos(pitch)

        if stage != 3:
            qpos[3], qpos[6] = np.cos(yaw / 2), np.sin(yaw / 2)

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
