import os
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import trackio
from gymnasium.wrappers import RecordVideo
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import Collector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    ParallelEnv,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

import paft_env  # noqa: F401

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
CHECKPOINT_PATH = "./checkpoints/paft_torchrl.pt"
VIDEO_DIR = "./videos/torchrl"

NUM_ENVS = 10
FRAMES_PER_BATCH = 2048
TOTAL_FRAMES = FRAMES_PER_BATCH * 4096
SUB_BATCH_SIZE = 128
NUM_EPOCHS = 4

LR = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEFF = 0.1
CRITIC_COEFF = 0.25
MAX_GRAD_NORM = 0.5

HIDDEN_SIZE = 256
MIN_SCALE = 0.5
LR_DECAY_POWER = 3  # power curve exponent: higher = stays high longer

LOG_INTERVAL = 12
VIDEO_INTERVAL = 24
EVAL_INTERVAL = 6


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────


def make_env(num_envs: int = 1) -> TransformedEnv:
    base = (
        ParallelEnv(num_envs, lambda: GymEnv("Paft-v0"))
        if num_envs > 1
        else GymEnv("Paft-v0")
    )
    return TransformedEnv(
        base,
        Compose(
            ObservationNorm(in_keys=["observation"]), DoubleToFloat(), StepCounter()
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────


def ortho_init(module: nn.Module, gain: float = np.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.constant_(module.bias, 0.0)


def make_policy(env: TransformedEnv) -> ProbabilisticActor:
    obs_dim, act_dim = (
        env.observation_spec["observation"].shape[-1],
        env.action_spec.shape[-1],
    )

    net = nn.Sequential(
        nn.Linear(obs_dim, HIDDEN_SIZE),
        nn.Tanh(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.Tanh(),
        nn.Linear(HIDDEN_SIZE, 2 * act_dim),
        NormalParamExtractor(scale_lb=MIN_SCALE, scale_mapping="softplus"),
    )
    net.apply(ortho_init)
    ortho_init(net[-2], gain=0.01)

    return ProbabilisticActor(
        module=TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        ),
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )


def make_value(env: TransformedEnv) -> ValueOperator:
    obs_dim = env.observation_spec["observation"].shape[-1]

    net = nn.Sequential(
        nn.Linear(obs_dim, HIDDEN_SIZE),
        nn.Tanh(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.Tanh(),
        nn.Linear(HIDDEN_SIZE, 1),
    )
    net.apply(ortho_init)
    ortho_init(net[-1], gain=1.0)

    return ValueOperator(module=net, in_keys=["observation"])


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation & Recording
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(
    policy: ProbabilisticActor, env: TransformedEnv, num_episodes: int = 5
) -> dict:
    rewards, lengths = [], []
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for _ in range(num_episodes):
            rollout = env.rollout(10000, policy)
            rewards.append(rollout["next", "reward"].sum().item())
            lengths.append(rollout["step_count"].max().item())
    return {
        "eval_reward_mean": np.mean(rewards),
        "eval_reward_std": np.std(rewards),
        "eval_length_mean": np.mean(lengths),
    }


def record_video(
    policy: ProbabilisticActor, loc: np.ndarray, scale: np.ndarray, tag: str
):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    raw_env = RecordVideo(
        gym.make("Paft-v0", render_mode="rgb_array"),
        VIDEO_DIR,
        name_prefix=tag,
        episode_trigger=lambda _: True,
    )
    obs, total_reward = raw_env.reset()[0], 0.0

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for _ in range(10000):
            obs_t = torch.from_numpy(
                ((obs - loc) / (scale + 1e-8)).astype(np.float32)
            ).unsqueeze(0)
            action = (
                policy(TensorDict({"observation": obs_t}, batch_size=[1]))["action"]
                .squeeze(0)
                .numpy()
            )
            obs, reward, term, trunc, _ = raw_env.step(action)
            total_reward += reward
            if term or trunc:
                break

    raw_env.close()
    print(f"  Video: {tag} | Reward: {total_reward:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    trackio.init(project="paft-torchrl")

    # Initialize observation normalization with single env
    print("Initializing observation normalization...")
    init_env = make_env()
    init_env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    obs_loc, obs_scale = (
        init_env.transform[0].loc.clone(),
        init_env.transform[0].scale.clone(),
    )
    init_env.close()

    # Create parallel training env and eval env
    print(f"Creating {NUM_ENVS} parallel environments...")
    env, eval_env = make_env(NUM_ENVS), make_env()
    for e in (env, eval_env):
        e.transform[0].loc, e.transform[0].scale = obs_loc.clone(), obs_scale.clone()

    print(
        f"Obs: {env.observation_spec['observation'].shape}, Act: {env.action_spec.shape}, Batch: {env.batch_size}"
    )

    # Networks and optimizer
    policy, value = make_policy(env), make_value(env)
    collector = Collector(
        env, policy, frames_per_batch=FRAMES_PER_BATCH, total_frames=TOTAL_FRAMES
    )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=FRAMES_PER_BATCH),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        gamma=GAMMA, lmbda=GAE_LAMBDA, value_network=value, average_gae=True
    )
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value,
        clip_epsilon=CLIP_EPSILON,
        entropy_bonus=True,
        entropy_coeff=ENTROPY_COEFF,
        critic_coeff=CRITIC_COEFF,
        loss_critic_type="smooth_l1",
    )
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=LR, eps=1e-5)
    num_batches = TOTAL_FRAMES // FRAMES_PER_BATCH
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, TOTAL_FRAMES, eta_min=1e-6)

    # Training loop
    logs, best_eval_reward = defaultdict(list), float("-inf")
    print(
        f"\nTraining PPO | {NUM_ENVS} envs | {TOTAL_FRAMES:,} frames | {num_batches} batches"
    )
    print("=" * 70)

    for batch_idx, data in enumerate(collector):
        frames = (batch_idx + 1) * FRAMES_PER_BATCH
        losses = defaultdict(list)

        for _ in range(NUM_EPOCHS):
            advantage_module(data)
            replay_buffer.extend(data.reshape(-1).cpu())

            for _ in range(FRAMES_PER_BATCH // SUB_BATCH_SIZE):
                batch = replay_buffer.sample(SUB_BATCH_SIZE)
                loss_vals = loss_module(batch)
                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                losses["loss_objective"].append(loss_vals["loss_objective"].item())
                losses["loss_critic"].append(loss_vals["loss_critic"].item())
                losses["loss_entropy"].append(-loss_vals["loss_entropy"].item())

        scheduler.step()

        # Logging
        reward, steps = (
            data["next", "reward"].mean().item(),
            data["step_count"].max().item(),
        )
        logs["reward"].append(reward)
        metrics = {
            "frames": frames,
            "reward": reward,
            "steps": steps,
            "pg_loss": np.mean(losses["loss_objective"]),
            "vf_loss": np.mean(losses["loss_critic"]),
            "entropy": np.mean(losses["loss_entropy"]),
            "lr": scheduler.get_last_lr()[0],
        }

        if (batch_idx + 1) % EVAL_INTERVAL == 0:
            eval_metrics = evaluate(policy, eval_env, num_episodes=5)
            metrics.update(eval_metrics)
            logs["eval_reward"].append(eval_metrics["eval_reward_mean"])
            best_eval_reward = max(best_eval_reward, eval_metrics["eval_reward_mean"])

        trackio.log(metrics)

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            eval_str = (
                f" | Eval: {logs['eval_reward'][-1]:.1f}" if logs["eval_reward"] else ""
            )
            print(
                f"Batch {batch_idx+1:4d}/{num_batches} | Frames: {frames:,} | R: {reward:.2f} | Steps: {steps:3.0f} | Ent: {metrics['entropy']:.3f}{eval_str}"
            )

        if (batch_idx + 1) % VIDEO_INTERVAL == 0:
            record_video(
                policy, obs_loc.numpy(), obs_scale.numpy(), f"batch_{batch_idx+1}"
            )

    # Cleanup and final eval
    obs_loc, obs_scale = env.transform[0].loc.clone(), env.transform[0].scale.clone()
    collector.shutdown()

    print("\nFinal evaluation...")
    final_env = make_env()
    final_env.transform[0].loc, final_env.transform[0].scale = obs_loc, obs_scale
    final_metrics = evaluate(policy, final_env, num_episodes=10)
    print(
        f"Final reward: {final_metrics['eval_reward_mean']:.1f} ± {final_metrics['eval_reward_std']:.1f}"
    )
    record_video(policy, obs_loc.numpy(), obs_scale.numpy(), "final")

    # Save
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(
        {
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "obs_loc": obs_loc,
            "obs_scale": obs_scale,
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved: {CHECKPOINT_PATH}")

    final_env.close()
    env.close()
    eval_env.close()

    trackio.finish()
    print(f"\n{'='*70}\nTraining complete! Best eval: {best_eval_reward:.1f}")


if __name__ == "__main__":
    train()
