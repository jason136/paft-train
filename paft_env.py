"""
PAFT Environment for RL training.

Single MuJoCo locomotion environment for the Mini TARS robot
with clean reward shaping and mild randomization.
"""

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

    Reward: forward velocity + heading alignment + alive bonus
    Observation: IMU sensors (10) + relative heading (2) = 12 dims
    Action: normalized [-1, 1] for 4 actuators
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        xml_path: Optional[str] = None,
        frame_skip: int = 5,
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

        # Actions: normalized to [-1, 1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self._action_scale = np.array([np.pi, 50.0, np.pi, 50.0], dtype=np.float32)

        # Observations: sensors(10) + relative_heading(2) = 12
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(12,), dtype=np.float32
        )

        self._target_heading = 0.0
        self._step_count = 0

    def _get_obs(self) -> np.ndarray:
        """Sensor readings + relative heading in local frame."""
        sensors = self.data.sensordata[:10].astype(np.float32)

        # Robot orientation quaternion
        qw, qx, qy, qz = self.data.qpos[3:7]

        # Rotation matrix (world → body)
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

        return np.concatenate([sensors, local_dir]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0)
        x_before, y_before = self.data.qpos[0], self.data.qpos[1]

        self.do_simulation(action * self._action_scale, self.frame_skip)
        self._step_count += 1

        x_after, y_after = self.data.qpos[0], self.data.qpos[1]
        qw, qx, qy, qz = self.data.qpos[3:7]

        # === REWARD ===
        # 1. Forward velocity toward target (DOMINANT objective)
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

        # Combined reward - forward motion is everything, no free alive bonus
        reward = (
            forward_vel * 100.0  # Strong: movement is the only way to get reward
            + alignment * 0.1  # Small bonus for facing right direction
        )

        # === TERMINATION ===
        up_z = 1 - 2 * (qx**2 + qy**2)
        tilt_angle = np.arccos(np.clip(up_z, -1, 1))
        terminated = bool(tilt_angle > np.radians(60))  # 60° tilt = fall

        # Penalty for falling
        if terminated:
            reward -= 10.0

        self._update_arrow()

        return self._get_obs(), reward, terminated, False, {"forward_vel": forward_vel}

    def reset_model(self) -> np.ndarray:
        self._step_count = 0

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

        # Randomize yaw (robot orientation)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        qpos[3], qpos[6] = np.cos(yaw / 2), np.sin(yaw / 2)

        # Arms in neutral forward position
        qpos[7] = qpos[9] = self.np_random.uniform(0.3, 0.8)

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
