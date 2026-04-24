import os
from collections import defaultdict
from functools import partial

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import wandb
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

NUM_ENVS = 50
FRAMES_PER_BATCH = 16384
TOTAL_FRAMES = FRAMES_PER_BATCH * 4096
SUB_BATCH_SIZE = 512
NUM_EPOCHS = 3

LR = 1e-4
GAMMA = 0.99
GAE_LAMBDA = 0.9
CLIP_EPSILON = 0.1
ENTROPY_COEFF = 0.05
CRITIC_COEFF = 0.25
MAX_GRAD_NORM = 0.5

HIDDEN_SIZE = 128
MIN_SCALE = 0.12
LR_DECAY_POWER = 3  # power curve exponent: higher = stays high longer

LOG_INTERVAL = 12
VIDEO_INTERVAL = 48
EVAL_INTERVAL = 6
CHECKPOINT_INTERVAL = 240


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────


def make_env(num_envs: int = 1) -> TransformedEnv:
    base = (
        ParallelEnv(num_envs, partial(GymEnv, "Paft-v0"))
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


def run_policy_episode(
    policy: ProbabilisticActor,
    loc: np.ndarray,
    scale: np.ndarray,
    record_tag: str | None = None,
    max_steps: int = 10000,
    start_stage: int | None = None,
) -> dict:
    """Run one deterministic episode in the raw Gym env."""
    if record_tag is None:
        env = gym.make("Paft-v0")
    else:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        env = RecordVideo(
            gym.make("Paft-v0", render_mode="rgb_array"),
            VIDEO_DIR,
            name_prefix=record_tag,
            episode_trigger=lambda _: True,
        )

    if start_stage is not None:
        env.unwrapped.set_start_stage(start_stage)

    obs, _ = env.reset()
    total_reward = 0.0
    steps_completed = 0
    transitions = 0
    max_stage = 0
    prev_stage = 0
    component_sums = defaultdict(float)

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for step in range(max_steps):
            obs_t = torch.from_numpy(
                ((obs - loc) / (scale + 1e-8)).astype(np.float32)
            ).unsqueeze(0)
            action = (
                policy(TensorDict({"observation": obs_t}, batch_size=[1]))["action"]
                .squeeze(0)
                .numpy()
            )
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            steps_completed = info.get("steps_completed", 0)
            stage = info.get("gait_stage", 0)
            if stage != prev_stage:
                transitions += 1
                prev_stage = stage
            max_stage = max(max_stage, stage)
            for k in ("r_velocity", "r_gait", "r_survive"):
                component_sums[k] += info.get(k, 0.0)
            if term or trunc:
                break

    env.close()
    n_steps = step + 1
    result = {
        "total_reward": total_reward,
        "steps": n_steps,
        "steps_completed": steps_completed,
        "transitions": transitions,
        "max_stage": max_stage,
    }
    for k, v in component_sums.items():
        result[k] = float(v)
    return result


def evaluate(
    policy: ProbabilisticActor,
    loc: np.ndarray,
    scale: np.ndarray,
) -> dict:
    episodes = [run_policy_episode(policy, loc, scale, start_stage=s) for s in range(4)]
    result = {
        "eval_reward_mean": np.mean([e["total_reward"] for e in episodes]),
        "eval_reward_std": np.std([e["total_reward"] for e in episodes]),
        "eval_length_mean": np.mean([e["steps"] for e in episodes]),
        "eval_transitions_mean": np.mean([e["transitions"] for e in episodes]),
    }
    for k in ("r_velocity", "r_gait", "r_survive"):
        result[f"eval_{k}"] = np.mean([e.get(k, 0.0) for e in episodes])
    return result


def record_video(
    policy: ProbabilisticActor, loc: np.ndarray, scale: np.ndarray, tag: str
):
    episode = run_policy_episode(policy, loc, scale, record_tag=tag)
    print(
        f"  Video: {tag} | R: {episode['total_reward']:.1f} | Steps: {episode['steps']} | "
        f"Trans: {episode['transitions']} | "
        f"vel={episode.get('r_velocity', 0):.1f} gait={episode.get('r_gait', 0):.1f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


def train(resume_path: str | None = None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    wandb.init(project="paft-torchrl")

    # Initialize observation normalization with single env
    print("Initializing observation normalization...")
    init_env = make_env()
    init_env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    obs_loc, obs_scale = (
        init_env.transform[0].loc.clone(),
        init_env.transform[0].scale.clone(),
    )
    init_env.close()

    # Create parallel training env
    print(f"Creating {NUM_ENVS} parallel environments...")
    env = make_env(NUM_ENVS)
    env.transform[0].loc, env.transform[0].scale = obs_loc.clone(), obs_scale.clone()

    print(
        f"Obs: {env.observation_spec['observation'].shape}, Act: {env.action_spec.shape}, Batch: {env.batch_size}"
    )

    # Networks
    policy, value = make_policy(env), make_value(env)

    if resume_path is not None:
        print(f"Resuming from {resume_path} (fresh optimizer + scheduler)...")
        ckpt = torch.load(resume_path, weights_only=False)
        policy.load_state_dict(ckpt["policy"])
        value.load_state_dict(ckpt["value"])
        if "obs_loc" in ckpt:
            obs_loc = ckpt["obs_loc"]
            obs_scale = ckpt["obs_scale"]
            env.transform[0].loc = obs_loc.clone()
            env.transform[0].scale = obs_scale.clone()
        print(f"  Loaded weights from batch {ckpt.get('batch_idx', '?')}")
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_batches, eta_min=1e-6
    )

    # Training loop
    logs, best_eval_reward = defaultdict(list), float("-inf")
    print(
        f"\nTraining PPO | {NUM_ENVS} envs | {TOTAL_FRAMES:,} frames | {num_batches} batches"
    )
    print("=" * 70)

    for batch_idx, data in enumerate(collector):
        frames = (batch_idx + 1) * FRAMES_PER_BATCH
        losses = defaultdict(list)

        advantage_module(data)
        replay_buffer.extend(data.reshape(-1).cpu())

        for _ in range(NUM_EPOCHS):
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
        reward = data["next", "reward"].mean().item()
        logs["reward"].append(reward)
        metrics = {
            "reward": reward,
            "pg_loss": np.mean(losses["loss_objective"]),
            "vf_loss": np.mean(losses["loss_critic"]),
            "entropy": np.mean(losses["loss_entropy"]),
            "lr": scheduler.get_last_lr()[0],
        }

        if (batch_idx + 1) % EVAL_INTERVAL == 0:
            eval_metrics = evaluate(policy, obs_loc.numpy(), obs_scale.numpy())
            metrics.update(eval_metrics)
            logs["eval_reward"].append(eval_metrics["eval_reward_mean"])
            best_eval_reward = max(best_eval_reward, eval_metrics["eval_reward_mean"])

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            wandb.log(metrics)

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            eval_str = (
                f" | Eval: {logs['eval_reward'][-1]:.1f}" if logs["eval_reward"] else ""
            )
            print(
                f"Batch {batch_idx+1:4d}/{num_batches} | R: {reward:.2f} | Ent: {metrics['entropy']:.3f}{eval_str}"
            )

        if (batch_idx + 1) % VIDEO_INTERVAL == 0:
            record_video(
                policy, obs_loc.numpy(), obs_scale.numpy(), f"batch_{batch_idx+1}"
            )

        if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
            ckpt_path = f"./checkpoints/batch_{batch_idx+1}.pt"
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "value": value.state_dict(),
                    "obs_loc": env.transform[0].loc.clone(),
                    "obs_scale": env.transform[0].scale.clone(),
                    "batch_idx": batch_idx + 1,
                },
                ckpt_path,
            )
            print(f"  Checkpoint saved: {ckpt_path}")

    # Cleanup and final eval
    obs_loc, obs_scale = env.transform[0].loc.clone(), env.transform[0].scale.clone()
    collector.shutdown()

    print("\nFinal evaluation...")
    final_metrics = evaluate(policy, obs_loc.numpy(), obs_scale.numpy())
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

    wandb.finish()
    print(f"\n{'='*70}\nTraining complete! Best eval: {best_eval_reward:.1f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume from (fresh optimizer/scheduler)",
    )
    args = parser.parse_args()
    train(resume_path=args.resume)
