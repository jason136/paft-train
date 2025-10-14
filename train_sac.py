"""
Minimal SAC training loop for MiniTars-v0 following TorchRL patterns
Based on: https://pytorch.org/rl/stable/tutorials/getting-started-5.html
"""
import torch
from torch.optim import Adam

from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq

from torchrl.envs import GymEnv, TransformedEnv, StepCounter
from torchrl.modules import MLP, ProbabilisticActor, NormalParamExtractor, TanhNormal, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.objectives import SACLoss, SoftUpdate
from torchrl.record import CSVLogger, VideoRecorder


def make_env():
    import mini_tars_env  # registers MiniTars-v0
    return TransformedEnv(GymEnv("MiniTars-v0"), StepCounter(max_steps=1000))


def make_sac_agent(env):
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    
    # Actor: obs -> (loc, scale) -> TanhNormal -> action
    actor_net = MLP(in_features=obs_dim, out_features=2 * act_dim, num_cells=[256, 256])
    actor_module = Mod(actor_net, in_keys=["observation"], out_keys=["loc_scale"])
    param_extractor = Mod(NormalParamExtractor(), in_keys=["loc_scale"], out_keys=["loc", "scale"])
    
    actor = ProbabilisticActor(
        Seq(actor_module, param_extractor),
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": env.action_spec.space.low, "high": env.action_spec.space.high},
        return_log_prob=True,
    )
    
    # Critic: (obs, action) -> Q-value
    qvalue_net = MLP(in_features=obs_dim + act_dim, out_features=1, num_cells=[256, 256])
    qvalue = Mod(qvalue_net, in_keys=["observation", "action"], out_keys=["state_action_value"])
    
    return actor, qvalue


def train(total_frames=50_000, frames_per_batch=1000, init_random_frames=5_000, batch_size=256):
    torch.manual_seed(0)
    
    env = make_env()
    env.set_seed(0)
    
    actor, qvalue = make_sac_agent(env)
    
    # SAC Loss with default keys (reads from collector output directly)
    loss_fn = SACLoss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=2,
        loss_function="smooth_l1",
    )
    loss_fn.make_value_estimator()
    
    # Optimizers
    actor_optim = Adam(loss_fn.actor_network_params.flatten_keys().values(), lr=3e-4)
    q_optim = Adam(loss_fn.qvalue_network_params.flatten_keys().values(), lr=3e-4)
    alpha_optim = Adam([loss_fn.log_alpha], lr=3e-4)
    
    # Target network updater
    target_updater = SoftUpdate(loss_fn, eps=0.995)
    
    # Data collector
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        init_random_frames=init_random_frames,
    )
    
    # Replay buffer (no transform—use collector output as-is)
    rb = ReplayBuffer(storage=LazyTensorStorage(200_000))
    
    # Training loop
    for i, data in enumerate(collector):
        rb.extend(data)
        
        if len(rb) < init_random_frames:
            continue
        
        for _ in range(frames_per_batch // batch_size):
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
        
        if i % 10 == 0:
            avg_reward = data["next", "reward"].mean().item()
            print(f"Iteration {i}, collected {data.numel()} frames, rb size {len(rb)}, avg reward: {avg_reward:.3f}")
    
    env.close()
    print("Training complete!")
    
    # Record video of trained policy
    print("\nRecording video...")
    logger = CSVLogger(exp_name="sac_minitars", log_dir="./logs", video_format="mp4")
    video_recorder = VideoRecorder(logger, tag="trained_policy")
    record_env = TransformedEnv(
        GymEnv("MiniTars-v0", from_pixels=True, pixels_only=False),
        video_recorder,
    )
    
    with torch.no_grad():
        record_env.rollout(max_steps=1000, policy=actor)
    
    video_recorder.dump()
    record_env.close()
    print(f"Video saved to ./logs/sac_minitars/videos/")


if __name__ == "__main__":
    train()
