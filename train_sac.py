import os
import torch

from torch.optim import Adam

from tensordict.nn import TensorDictModule, TensorDictSequential

from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import DoubleToFloat, StepCounter, ObservationNorm, RewardScaling

from torchrl.modules import (
    MLP,
    ProbabilisticActor,
    NormalParamExtractor,
    TanhNormal,
    ValueOperator,
)

from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.objectives.sac import SACLoss


def make_env(from_pixels: bool = False) -> TransformedEnv:
    # MiniTars-v0 is registered on import from mini_tars_env
    import mini_tars_env  # noqa: F401

    base = GymEnv("MiniTars-v0")
    env = TransformedEnv(
        base,
        DoubleToFloat(),
        StepCounter(max_steps=1000),
    )
    return env


def build_sac(env: TransformedEnv):
    obs_key = "observation"
    action_spec = env.action_spec
    obs_spec = env.observation_spec

    # TorchRL GymEnv returns a CompositeSpec; pull the primary observation
    obs_dim = obs_spec[obs_key].shape[-1]
    act_dim = action_spec.shape[-1]

    # Policy network -> Normal params -> Tanh squashed
    policy_backbone = MLP(in_features=obs_dim, out_features=2 * act_dim, num_cells=[256, 256])
    policy_td = TensorDictModule(policy_backbone, in_keys=[obs_key], out_keys=["loc_and_scale"])
    extractor = NormalParamExtractor()
    extractor_td = TensorDictModule(extractor, in_keys=["loc_and_scale"], out_keys=["loc", "scale"])

    def dist_module(td):
        return TanhNormal(td["loc"], td["scale"], low=action_spec.space.low, high=action_spec.space.high)

    policy = ProbabilisticActor(
        TensorDictSequential(policy_td, extractor_td),
        in_keys=[obs_key],
        spec=action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": torch.as_tensor(action_spec.space.low, dtype=torch.float32),
            "high": torch.as_tensor(action_spec.space.high, dtype=torch.float32),
        },
    )

    # Critics: one Q module that outputs two Q-values on the last dim
    q_backbone = MLP(in_features=obs_dim + act_dim, out_features=2, num_cells=[256, 256])
    q_module = ValueOperator(q_backbone, in_keys=[obs_key, "action"], out_keys=["state_action_value"])

    loss_module = SACLoss(
        actor_network=policy,
        qvalue_network=q_module,
        value_network=None,
        fixed_alpha=True,
        target_entropy="auto",
        delay_qvalue=True,
        delay_value=True,
        num_qvalue_nets=2,
    )

    actor_optim = Adam(loss_module.actor_network.parameters(), lr=3e-4)
    q_optim = Adam(loss_module.qvalue_network.parameters(), lr=3e-4)

    return loss_module, actor_optim, q_optim, policy


def train(seed: int = 0, total_frames: int = 50_000, frames_per_batch: int = 1000, init_random_frames: int = 5_000):
    torch.manual_seed(seed)
    env = make_env()
    env.set_seed(seed)

    loss_module, actor_optim, q_optim, policy = build_sac(env)

    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        init_random_frames=init_random_frames,
    )

    rb = ReplayBuffer(storage=LazyTensorStorage(200_000))

    batch_size = 256
    for i, data in enumerate(collector):
        rb.extend(data)
        if len(rb) < init_random_frames:
            continue
        for _ in range(frames_per_batch // batch_size):
            sample = rb.sample(batch_size)
            loss_vals = loss_module(sample)

            q_optim.zero_grad()
            loss_vals["loss_qvalue"].backward(retain_graph=True)
            q_optim.step()

            actor_optim.zero_grad()
            loss_vals["loss_actor"].backward()
            actor_optim.step()

    env.close()


if __name__ == "__main__":
    train()


