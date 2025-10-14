import os
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils import EzPickle


class MiniTarsMujocoEnv(MujocoEnv, EzPickle):
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

        EzPickle.__init__(self, xml_path=xml_path, frame_skip=frame_skip, domain_randomize=domain_randomize, render_mode=render_mode)

        self._domain_randomize = domain_randomize

        super().__init__(
            model_path=xml_path,
            frame_skip=frame_skip,
            observation_space=None,
            render_mode=render_mode,
        )

        # Define action space from number of actuators (position servos)
        action_dim = self.model.nu
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(action_dim,), dtype=np.float32)

        # Observation space: qpos + qvel
        obs_dim = self.model.nq + self.model.nv
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._prev_x: Optional[float] = None

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.ravel()
        qvel = self.data.qvel.ravel()
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xpos_before = float(self.data.qpos[0])  # world x position
        self.do_simulation(action, self.frame_skip)
        xpos_after = float(self.data.qpos[0])
        x_velocity = (xpos_after - xpos_before) / self.dt

        # Reward shaping similar to standard MuJoCo locomotion tasks
        forward_reward = x_velocity
        ctrl_cost = 1e-2 * float(np.sum(np.square(action)))
        reward = forward_reward - ctrl_cost

        observation = self._get_obs()
        terminated = False
        # Early termination if the torso falls below a threshold
        z_height = float(self.data.qpos[2])
        if z_height < 0.05:
            terminated = True

        info: Dict = {"x_velocity": x_velocity, "forward_reward": forward_reward, "ctrl_cost": ctrl_cost}
        truncated = False
        return observation, reward, terminated, truncated, info

    def reset_model(self) -> np.ndarray:
        if self._domain_randomize:
            self._randomize_domain()

        # Small noise around default pose
        qpos = self.init_qpos + self.np_random.uniform(low=-1e-2, high=1e-2, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-5e-3, high=5e-3, size=self.model.nv)
        self.set_state(qpos, qvel)
        self._prev_x = float(self.data.qpos[0])
        return self._get_obs()

    def _randomize_domain(self) -> None:
        # Example domain randomization: scale body masses and friction
        mass_scale = self.np_random.uniform(0.8, 1.2)
        self.model.body_mass[:] *= mass_scale
        friction_scale = self.np_random.uniform(0.8, 1.2)
        self.model.geom_friction[:, 0] *= friction_scale  # sliding friction


def _make_env(**kwargs) -> gym.Env:
    return MiniTarsMujocoEnv(**kwargs)


def register_env() -> None:
    gym.register(
        id="MiniTars-v0",
        entry_point="mini_tars_env:MiniTarsMujocoEnv",
        max_episode_steps=1000,
    )


# Allow import side-effect registration if desired
try:
    register_env()
except Exception:
    # If already registered, ignore
    pass


