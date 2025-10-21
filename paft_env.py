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

        # Observation space: qpos + qvel + target_direction (2D unit vector)
        obs_dim = self.model.nq + self.model.nv + 2
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._prev_x: Optional[float] = None
        self._target_heading: float = 0.0  # target angle in radians

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.ravel()
        qvel = self.data.qvel.ravel()
        # Add target direction as 2D unit vector (cos, sin)
        target_dir = np.array(
            [np.cos(self._target_heading), np.sin(self._target_heading)],
            dtype=np.float32,
        )
        return np.concatenate([qpos, qvel, target_dir]).astype(np.float32)

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

        # Stability: penalize if robot is too low (falling)
        z_height = float(self.data.qpos[2])
        height_penalty = -5.0 if z_height < 0.10 else 0.0  # Penalty for falling

        # REMOVE height bonus - standing still should NOT be rewarded
        height_bonus = 1.0 if z_height > 0.15 else 0.0

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
            + height_penalty
            + 0.1 * height_bonus
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

        # Much more aggressive initial randomization to help discover gaits
        qpos = self.init_qpos.copy()

        # Keep base position roughly stable but allow some variation
        qpos[0] += self.np_random.uniform(-0.1, 0.1)  # x position
        qpos[1] += self.np_random.uniform(-0.1, 0.1)  # y position
        qpos[2] += self.np_random.uniform(-0.02, 0.05)  # z height variation

        # Keep body mostly upright - only small rotations around Z axis (yaw)
        # This ensures arms stay in their valid planes of rotation
        yaw = self.np_random.uniform(-0.2, 0.2)  # ±11 degrees rotation around Z
        # Quaternion for rotation around Z axis: [cos(θ/2), 0, 0, sin(θ/2)]
        qpos[3] = np.cos(yaw / 2)  # w
        qpos[4] = 0.0  # x
        qpos[5] = 0.0  # y
        qpos[6] = np.sin(yaw / 2)  # z

        # Randomize arm angles significantly to explore gaits
        # Arms can only move in their defined planes of rotation
        # Joint order: left_swing, left_shrug, right_swing, right_shrug
        # Start from neutral with safe variations to prevent body clipping

        # Swing joints: rotation around X axis (forward/back motion) - ANGULAR (radians)
        qpos[7] = self.np_random.uniform(-1.2, 1.2)  # left swing: ±69 degrees
        qpos[9] = self.np_random.uniform(-1.2, 1.2)  # right swing: ±69 degrees

        # Shrug joints: LINEAR slide up/down along Z axis - POSITION (meters)
        qpos[8] = self.np_random.uniform(
            -0.03, 0.03
        )  # left shrug: ±3cm vertical motion
        qpos[10] = self.np_random.uniform(
            -0.03, 0.03
        )  # right shrug: ±3cm vertical motion

        # Give initial velocities to encourage movement exploration
        qvel = self.init_qvel.copy()
        qvel[0] = self.np_random.uniform(-0.5, 0.5)  # x velocity
        qvel[1] = self.np_random.uniform(-0.5, 0.5)  # y velocity

        # Add rotational momentum - robot starts spinning
        qvel[3] = self.np_random.uniform(-2.0, 2.0)  # angular velocity around x (roll)
        qvel[4] = self.np_random.uniform(-2.0, 2.0)  # angular velocity around y (pitch)
        qvel[5] = self.np_random.uniform(
            -3.0, 3.0
        )  # angular velocity around z (yaw) - stronger

        # Arms start moving wildly
        qvel[6] = self.np_random.uniform(
            -4.0, 4.0
        )  # left swing angular velocity - INCREASED
        qvel[7] = self.np_random.uniform(
            -3.0, 3.0
        )  # left shrug angular velocity - INCREASED
        qvel[8] = self.np_random.uniform(
            -4.0, 4.0
        )  # right swing angular velocity - INCREASED
        qvel[9] = self.np_random.uniform(
            -3.0, 3.0
        )  # right shrug angular velocity - INCREASED

        self.set_state(qpos, qvel)
        self._prev_x = float(self.data.qpos[0])
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
