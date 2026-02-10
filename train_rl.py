"""
PPO Training for PAFT Robot (from scratch)

Clean, minimal PPO implementation with:
- Orthogonal initialization
- Running observation normalization
- Vectorized environments
- Proper GAE computation

Usage:
    python train_rl.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import trackio
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.vector import SyncVectorEnv

import paft_env  # noqa: F401

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
CHECKPOINT_PATH = "./checkpoints/paft_ppo.pt"
VIDEO_DIR = "./videos/rl"

# Training
NUM_ENVS = 4
TOTAL_STEPS = 2_000_000
STEPS_PER_ENV = 512  # Steps per env per rollout
BATCH_SIZE = NUM_ENVS * STEPS_PER_ENV  # 2048

# PPO
NUM_EPOCHS = 4
MINIBATCH_SIZE = 128
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.25  # Lower value coefficient - don't let bad value estimates dominate
MAX_GRAD_NORM = 0.5

# Network
HIDDEN_SIZE = 256
LOG_STD_INIT = 0.0
LOG_STD_MIN = -1.0
LOG_STD_MAX = 1.0

# Observation normalization
OBS_NORM_STEPS = 100_000  # Freeze obs normalization after this many steps

# Logging
VIDEO_INTERVAL = 50
LOG_INTERVAL = 5


# ─────────────────────────────────────────────────────────────────────────────
# Orthogonal Initialization
# ─────────────────────────────────────────────────────────────────────────────


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for better gradient flow."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ─────────────────────────────────────────────────────────────────────────────
# Running Statistics for Observation Normalization
# ─────────────────────────────────────────────────────────────────────────────


class RunningMeanStd:
    """Welford's online algorithm for running mean/std."""

    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Actor-Critic Network
# ─────────────────────────────────────────────────────────────────────────────


class ActorCritic(nn.Module):
    """Separate actor and critic networks with proper initialization."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = HIDDEN_SIZE):
        super().__init__()

        # Shared feature extractor? No - separate is more stable for PPO

        # Actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),  # Small init for actions
        )
        self._log_std = nn.Parameter(torch.full((act_dim,), LOG_STD_INIT))

        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    @property
    def log_std(self) -> torch.Tensor:
        return torch.clamp(self._log_std, LOG_STD_MIN, LOG_STD_MAX)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None):
        mean = self.actor(obs)
        std = self.log_std.exp()
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)

        return action, log_prob, entropy, value


# ─────────────────────────────────────────────────────────────────────────────
# Environment Setup
# ─────────────────────────────────────────────────────────────────────────────


def make_env(seed: int):
    """Factory function for vectorized envs."""

    def _init():
        env = gym.make("Paft-v0")
        env.reset(seed=seed)
        return env

    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Video Recording
# ─────────────────────────────────────────────────────────────────────────────


def record_video(agent: ActorCritic, obs_rms: RunningMeanStd, tag: str):
    """Record evaluation video."""
    os.makedirs(VIDEO_DIR, exist_ok=True)

    env = gym.make("Paft-v0", render_mode="rgb_array")
    env = RecordVideo(env, VIDEO_DIR, name_prefix=tag, episode_trigger=lambda _: True)

    obs, _ = env.reset()
    total_reward = 0.0

    for _ in range(1000):
        obs_norm = obs_rms.normalize(obs).astype(np.float32)
        with torch.no_grad():
            action = agent.actor(torch.from_numpy(obs_norm).unsqueeze(0))
        action = action.squeeze().numpy()
        action = np.clip(action, -1.0, 1.0)  # Match training: clip, not tanh

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    env.close()
    print(f"  ✓ Video: {tag} | Reward: {total_reward:.1f}")
    return total_reward


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Initialize trackio
    trackio.init(project="paft-rl")

    # Vectorized environments
    envs = SyncVectorEnv([make_env(SEED + i) for i in range(NUM_ENVS)])
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    # Agent
    agent = ActorCritic(obs_dim, act_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    # Observation normalization
    obs_rms = RunningMeanStd(shape=(obs_dim,))

    # Storage
    obs_buf = np.zeros((STEPS_PER_ENV, NUM_ENVS, obs_dim), dtype=np.float32)
    act_buf = np.zeros((STEPS_PER_ENV, NUM_ENVS, act_dim), dtype=np.float32)
    logp_buf = np.zeros((STEPS_PER_ENV, NUM_ENVS), dtype=np.float32)
    rew_buf = np.zeros((STEPS_PER_ENV, NUM_ENVS), dtype=np.float32)
    done_buf = np.zeros((STEPS_PER_ENV, NUM_ENVS), dtype=np.float32)
    val_buf = np.zeros((STEPS_PER_ENV, NUM_ENVS), dtype=np.float32)

    # Training state
    global_step = 0
    num_updates = TOTAL_STEPS // BATCH_SIZE
    episode_rewards = []
    episode_lengths = []
    current_ep_rewards = np.zeros(NUM_ENVS)
    current_ep_lengths = np.zeros(NUM_ENVS)

    obs, _ = envs.reset(seed=SEED)

    print(f"Training PPO from scratch")
    print(f"  Envs: {NUM_ENVS} | Steps: {TOTAL_STEPS:,} | Updates: {num_updates}")
    print("=" * 60)

    for update in range(num_updates):
        # === ROLLOUT ===
        for step in range(STEPS_PER_ENV):
            global_step += NUM_ENVS

            # Normalize observations
            obs_norm = obs_rms.normalize(obs).astype(np.float32)
            obs_buf[step] = obs_norm

            with torch.no_grad():
                obs_t = torch.from_numpy(obs_norm)
                action, logp, _, value = agent.get_action_and_value(obs_t)
                action = action.numpy()
                logp = logp.numpy()
                value = value.numpy()

            # Clip actions
            action = np.clip(action, -1.0, 1.0)
            act_buf[step] = action
            logp_buf[step] = logp
            val_buf[step] = value

            # Step environments
            next_obs, reward, terminated, truncated, infos = envs.step(action)
            done = np.logical_or(terminated, truncated)

            rew_buf[step] = reward
            done_buf[step] = done

            # Track episodes
            current_ep_rewards += reward
            current_ep_lengths += 1

            for i, d in enumerate(done):
                if d:
                    episode_rewards.append(current_ep_rewards[i])
                    episode_lengths.append(current_ep_lengths[i])
                    current_ep_rewards[i] = 0
                    current_ep_lengths[i] = 0

            # Update observation stats (freeze after OBS_NORM_STEPS)
            if global_step < OBS_NORM_STEPS:
                obs_rms.update(next_obs)
            obs = next_obs

        # === COMPUTE ADVANTAGES ===
        with torch.no_grad():
            next_obs_norm = obs_rms.normalize(obs).astype(np.float32)
            next_value = agent.get_value(torch.from_numpy(next_obs_norm)).numpy()

        # GAE
        advantages = np.zeros_like(rew_buf)
        lastgae = 0
        for t in reversed(range(STEPS_PER_ENV)):
            if t == STEPS_PER_ENV - 1:
                next_values = next_value
            else:
                next_values = val_buf[t + 1]

            next_nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + GAMMA * next_values * next_nonterminal - val_buf[t]
            advantages[t] = lastgae = (
                delta + GAMMA * GAE_LAMBDA * next_nonterminal * lastgae
            )

        returns = advantages + val_buf

        # Flatten batch
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1, act_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        # === PPO UPDATE with KL early stopping ===
        b_inds = np.arange(BATCH_SIZE)
        clipfracs = []
        approx_kls = []

        for epoch in range(NUM_EPOCHS):
            np.random.shuffle(b_inds)

            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, new_logp, entropy, new_value = agent.get_action_and_value(
                    torch.from_numpy(b_obs[mb_inds]),
                    torch.from_numpy(b_act[mb_inds]),
                )

                # Advantage normalization
                mb_adv = torch.from_numpy(b_adv[mb_inds])
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss
                logratio = new_logp - torch.from_numpy(b_logp[mb_inds])
                ratio = logratio.exp()

                # Approximate KL divergence for early stopping
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    approx_kls.append(approx_kl)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped for stability)
                v_loss = (
                    0.5 * ((new_value - torch.from_numpy(b_ret[mb_inds])) ** 2).mean()
                )

                # Entropy loss
                ent_loss = entropy.mean()

                # Total loss
                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                # Track clip fraction
                with torch.no_grad():
                    clipfracs.append(
                        ((ratio - 1.0).abs() > CLIP_EPS).float().mean().item()
                    )

        # === LOGGING ===
        if len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
        else:
            avg_reward = 0.0
            avg_length = 0.0

        std_dev = agent.log_std.exp().mean().item()
        mean_kl = np.mean(approx_kls)

        metrics = {
            "global_step": global_step,
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "pg_loss": pg_loss.item(),
            "vf_loss": v_loss.item(),
            "entropy": ent_loss.item(),
            "std_dev": std_dev,
            "approx_kl": mean_kl,
            "clip_frac": np.mean(clipfracs),
            "lr": optimizer.param_groups[0]["lr"],
        }
        trackio.log(metrics)

        if (update + 1) % LOG_INTERVAL == 0:
            print(
                f"Update {update+1:4d}/{num_updates} | "
                f"Steps: {global_step:,} | "
                f"Reward: {avg_reward:6.1f} | "
                f"Len: {avg_length:5.0f} | "
                f"KL: {mean_kl:.4f} | "
                f"Std: {std_dev:.3f}"
            )

        # Record video
        if (update + 1) % VIDEO_INTERVAL == 0:
            record_video(agent, obs_rms, f"update_{update+1}")

    envs.close()

    # Final video
    print("\nRecording final video...")
    final_reward = record_video(agent, obs_rms, "final")

    # Save checkpoint
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "obs_mean": obs_rms.mean,
            "obs_var": obs_rms.var,
        },
        CHECKPOINT_PATH,
    )
    print(f"✓ Saved checkpoint: {CHECKPOINT_PATH}")

    trackio.finish()
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final evaluation reward: {final_reward:.1f}")
    if len(episode_rewards) > 0:
        print(f"Best avg reward (100 ep): {max(episode_rewards[-100:]):.1f}")


if __name__ == "__main__":
    train()
