import torch
from torch.optim import Adam

from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq

from torchrl.envs import GymEnv, TransformedEnv, StepCounter
from torchrl.modules import (
    MLP,
    ProbabilisticActor,
    NormalParamExtractor,
    TanhNormal,
    ValueOperator,
)
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.objectives import SACLoss, SoftUpdate
from torchrl.record import CSVLogger, VideoRecorder


def make_env():
    import paft_env

    return TransformedEnv(GymEnv("Paft-v0"), StepCounter(max_steps=250))


def make_sac_agent(env):
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]

    # Actor: obs -> (loc, scale) -> TanhNormal -> action
    actor_net = MLP(in_features=obs_dim, out_features=2 * act_dim, num_cells=[256, 256])
    actor_module = Mod(actor_net, in_keys=["observation"], out_keys=["loc_scale"])
    param_extractor = Mod(
        NormalParamExtractor(), in_keys=["loc_scale"], out_keys=["loc", "scale"]
    )

    actor = ProbabilisticActor(
        Seq(actor_module, param_extractor),
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
    )

    # Critic: (obs, action) -> Q-value
    qvalue_net = MLP(
        in_features=obs_dim + act_dim, out_features=1, num_cells=[256, 256]
    )
    qvalue = Mod(
        qvalue_net, in_keys=["observation", "action"], out_keys=["state_action_value"]
    )

    return actor, qvalue


def train(
    total_frames=10_000_000,  # Much more training
    frames_per_batch=2000,
    init_random_frames=50_000,  # Less random, more learning
    batch_size=256,
):
    torch.manual_seed(0)

    env = make_env()
    env.set_seed(0)

    actor, qvalue = make_sac_agent(env)

    loss_fn = SACLoss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=2,
        loss_function="smooth_l1",
        target_entropy=-1.0,  # HIGH entropy = MORE exploration
    )
    loss_fn.make_value_estimator()

    actor_optim = Adam(loss_fn.actor_network_params.flatten_keys().values(), lr=5e-4)
    q_optim = Adam(loss_fn.qvalue_network_params.flatten_keys().values(), lr=5e-4)
    alpha_optim = Adam([loss_fn.log_alpha], lr=5e-4)

    target_updater = SoftUpdate(loss_fn, eps=0.995)

    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        init_random_frames=init_random_frames,
    )

    rb = ReplayBuffer(storage=LazyTensorStorage(500_000))

    logger = CSVLogger(exp_name="sac_paft", log_dir="./logs", video_format="mp4")

    best_reward = float("-inf")
    total_frames_collected = 0

    for i, data in enumerate(collector):
        rb.extend(data)
        total_frames_collected += data.numel()

        if len(rb) < init_random_frames:
            print(
                f"Collecting random exploration data: {len(rb)}/{init_random_frames} frames"
            )
            continue

        num_updates = frames_per_batch // batch_size
        for update_idx in range(num_updates):
            sample = rb.sample(batch_size)

            # Update Q-networks
            loss_vals = loss_fn(sample)
            q_optim.zero_grad()
            loss_vals["loss_qvalue"].backward()
            q_optim.step()

            # Update actor (recompute after Q update)
            loss_vals = loss_fn(sample)
            actor_optim.zero_grad()
            loss_vals["loss_actor"].backward()
            actor_optim.step()

            # Update alpha (recompute after actor update)
            loss_vals = loss_fn(sample)
            alpha_optim.zero_grad()
            loss_vals["loss_alpha"].backward()
            alpha_optim.step()

            # Update target networks
            target_updater.step()

        avg_reward = data["next", "reward"].mean().item()
        max_reward = data["next", "reward"].max().item()
        min_reward = data["next", "reward"].min().item()

        if avg_reward > best_reward:
            best_reward = avg_reward

        alpha_value = loss_fn.log_alpha.exp().item()

        print(f"\n{'='*70}")
        print(
            f"Iteration {i} | Frames: {total_frames_collected:,}/{total_frames:,} ({100*total_frames_collected/total_frames:.1f}%)"
        )
        print(f"Replay Buffer: {len(rb):,} samples")
        print(
            f"Reward - Avg: {avg_reward:.3f} | Best: {best_reward:.3f} | Range: [{min_reward:.3f}, {max_reward:.3f}]"
        )
        print(f"Exploration (alpha): {alpha_value:.4f}")
        print(
            f"Loss - Actor: {loss_vals['loss_actor'].item():.4f} | Q-value: {loss_vals['loss_qvalue'].item():.4f}"
        )

        # Detailed info every 10 iterations
        if i % 10 == 0 and i > 0:
            info_keys = data.get("next", {}).get("info", {})
            if info_keys:
                print(f"\nEnvironment Info:")
                for key in [
                    "velocity_toward_target",
                    "forward_speed",
                    "velocity_perpendicular",
                    "z_height",
                    "leg_activity",
                ]:
                    if key in info_keys:
                        val = info_keys[key].mean().item()
                        print(f"  {key}: {val:.4f}")

            print(f"\nRecording video at iteration {i}...")
            video_recorder = VideoRecorder(logger, tag=f"training_iter_{i}")
            record_env = TransformedEnv(
                GymEnv("Paft-v0", from_pixels=True, pixels_only=False),
                video_recorder,
            )

            with torch.no_grad():
                record_env.rollout(max_steps=1000, policy=actor)

            video_recorder.dump()
            record_env.close()
            print(f"Video saved for iteration {i}")

    env.close()
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best average reward achieved: {best_reward:.3f}")
    print("=" * 70)

    # Record final video of trained policy
    print("\nRecording final video...")
    video_recorder = VideoRecorder(logger, tag="final_policy")
    record_env = TransformedEnv(
        GymEnv("Paft-v0", from_pixels=True, pixels_only=False),
        video_recorder,
    )

    with torch.no_grad():
        record_env.rollout(max_steps=1000, policy=actor)

    video_recorder.dump()
    record_env.close()
    print(f"All videos saved to ./logs/sac_paft/videos/")


if __name__ == "__main__":
    train()
