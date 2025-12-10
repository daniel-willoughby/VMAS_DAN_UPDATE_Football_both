"""
VMAS Football Adversarial Training
==================================

Trains two teams of agents simultaneously, red and blue, within the VMAS Football environment.

The agents can be trained from initial random weights and biases or from existing saved checkpoints.
"""


# Imports

import torch
from torch import multiprocessing
from torch.distributions import Categorical

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from tensordict.nn import set_composite_lp_aggregate, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor

from matplotlib import pyplot as plt

from tqdm import tqdm

# Module-level constants and configuration
BLUE = 0                                    # Team 0 is the blue team
RED = 1                                     # Team 1 is the red team
LOAD_CHECKPOINT = False                     # Load pre-saved weights and biases?
vmas_device = torch.device("cpu")           # we do not use a GPU

# Configurable Hyperparameters
frames_per_batch = 40_000
n_iters = 10000
total_frames = frames_per_batch * n_iters
num_epochs = 10
minibatch_size = 2000
lr_blue = 3e-4
lr_red = 3e-4
max_grad_norm = 0.5
clip_epsilon = 0.2
gamma = 0.99
lmbda = 0.95
entropy_eps = 0.01
share_parameters_policy = True              # team members share a single policy
set_composite_lp_aggregate(False).set()     # torchRL - disables auto log-probability aggregation
share_parameters_critic = True
mappo = True                                # IPPO if False


# Environment Parameters
n_blue_agents = 5
n_red_agents = 5
max_steps = 600
num_vmas_envs = (
    frames_per_batch // max_steps
)

env = VmasEnv(
    scenario="football",
    num_envs=num_vmas_envs,
    continuous_actions=False,
    max_steps=max_steps,
    device=vmas_device,
    n_blue_agents=n_blue_agents,
    n_red_agents=n_red_agents,
    ai_blue_agents=False,
    ai_red_agents=False,
    physically_different=False,
    enable_shooting=False
)


# Initialisation

torch.manual_seed(0)

env = TransformedEnv(
    env,
    RewardSum(in_keys=env.reward_keys, out_keys=[("agent_blue", "episode_reward"), ("agent_red", "episode_reward")]),
)

check_env_specs(env)

policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agent_blue", "observation"].shape[-1],
        n_agent_outputs=9,
        n_agents=n_blue_agents,
        centralised=False,
        share_params=share_parameters_policy,
        device=vmas_device,
        depth=2,
        num_cells=128,
        activation_class=torch.nn.Tanh,
    ),
)

policy_net2 = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agent_red", "observation"].shape[-1],
        n_agent_outputs=9,
        n_agents=n_red_agents,
        centralised=False,
        share_params=share_parameters_policy,
        device=vmas_device,
        depth=2,
        num_cells=128,
        activation_class=torch.nn.Tanh,
    ),
)

policy_module = TensorDictModule(
    policy_net,
    in_keys=[("agent_blue", "observation")],
    out_keys=[("agent_blue", "logits")],
)

policy_module2 = TensorDictModule(
    policy_net2,
    in_keys=[("agent_red", "observation")],
    out_keys=[("agent_red", "logits")],
)

policy = ProbabilisticActor(
    module=policy_module,
    # spec=env.action_spec_unbatched,
    in_keys=[("agent_blue", "logits")],
    out_keys=env.action_keys[0],
    distribution_class=Categorical,
    distribution_kwargs={},
    return_log_prob=True,
)

policy2 = ProbabilisticActor(
    module=policy_module2,
    # spec=env.action_spec_unbatched,
    in_keys=[("agent_red", "logits")],
    out_keys=env.action_keys[1],
    distribution_class=Categorical,
    distribution_kwargs={},
    return_log_prob=True,
)

critic_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agent_blue", "observation"].shape[-1],
    n_agent_outputs=1,
    n_agents=n_blue_agents,
    centralised=mappo,
    share_params=share_parameters_critic,
    device=vmas_device,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
)

critic_net2 = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agent_red", "observation"].shape[-1],
    n_agent_outputs=1,
    n_agents=n_red_agents,
    centralised=mappo,
    share_params=share_parameters_critic,
    device=vmas_device,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
)

critic = TensorDictModule(
    module=critic_net,
    in_keys=[("agent_blue", "observation")],
    out_keys=[("agent_blue", "state_value")],
)

critic2 = TensorDictModule(
    module=critic_net2,
    in_keys=[("agent_red", "observation")],
    out_keys=[("agent_red", "state_value")],
)

if LOAD_CHECKPOINT:
    policy = torch.load("policy_blue.pt", weights_only= False)
    policy2 = torch.load("policy_red.pt", weights_only= False)

combined_policy = TensorDictSequential(
    policy,
    policy2,
)

collector = SyncDataCollector(
    env,
    combined_policy,
    device=vmas_device,
    storing_device=vmas_device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

# Replay buffer creation

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=vmas_device
    ),  # We store the frames_per_batch collected at each iteration
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,  # We will sample minibatches of this size
)

replay_buffer2 = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=vmas_device
    ),  # We store the frames_per_batch collected at each iteration
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,  # We will sample minibatches of this size
)

# Loss function

loss_module = ClipPPOLoss(
    actor_network=combined_policy[BLUE],
    critic_network=critic,
    clip_epsilon=clip_epsilon,
    entropy_coeff=entropy_eps,
    normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
)

loss_module2 = ClipPPOLoss(
    actor_network=combined_policy[RED],
    critic_network=critic2,
    clip_epsilon=clip_epsilon,
    entropy_coeff=entropy_eps,
    normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
)

loss_module.set_keys(  # We have to tell the loss where to find the keys
    reward=env.reward_keys[0],
    action=env.action_keys[0],
    value=("agent_blue", "state_value"),
    done=("agent_blue", "done"),
    terminated=("agent_blue", "terminated"),
)

loss_module2.set_keys(  # We have to tell the loss where to find the keys
    reward=env.reward_keys[1],
    action=env.action_keys[1],
    value=("agent_red", "state_value"),
    done=("agent_red", "done"),
    terminated=("agent_red", "terminated"),
)

# we think gamma and lmbda are parameters for the estimator (GAE)

loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
)  # We build GAE

loss_module2.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
)  # We build GAE2


GAE = loss_module.value_estimator
GAE2 = loss_module2.value_estimator

optim = torch.optim.Adam(loss_module.parameters(), lr_blue)
optim2 = torch.optim.Adam(loss_module2.parameters(), lr_red)

pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

episode_reward_mean_list = []
episode_reward_mean_list2 = []

tuple_counter = 0

for tensordict_data in collector:
    tensordict_data.set(
        ("next", "agent_blue", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_keys[0]))),
    )
    tensordict_data.set(
        ("next", "agent_blue", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_keys[0]))),
    )
    tensordict_data.set(
        ("next", "agent_red", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_keys[1]))),
    )
    tensordict_data.set(
        ("next", "agent_red", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_keys[1]))),
    )

    with torch.no_grad():
        GAE(
            tensordict_data[BLUE],
            params=loss_module.critic_network_params,
            target_params=loss_module.target_critic_network_params,
        ) # Compute GAE and add it to the tensordict

    scrubbed = tensordict_data.exclude("agent_red", ("next", "agent_red"))
    data_view = scrubbed.reshape(-1)
    replay_buffer.extend(data_view)  # dw this appends "data_view" to the replay buffer

    with torch.no_grad():
        GAE2(
            tensordict_data[RED],
            params=loss_module2.critic_network_params,
            target_params=loss_module2.target_critic_network_params,
        )

    scrubbed = tensordict_data.exclude("agent_blue", ("next", "agent_blue"))
    data_view = scrubbed.reshape(-1)
    replay_buffer2.extend(data_view)  # DW this appends "data_view" to the replay buffer

    for _ in range(num_epochs):

        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            # print("blue subdata",subdata)
            loss_vals = loss_module(subdata)
            loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )  # Optional

            optim.step()
            optim.zero_grad()

        for _ in range(frames_per_batch // minibatch_size):
            subdata2 = replay_buffer2.sample()
            # print("red subdata",subdata)
            # print(subdata)
            loss_vals = loss_module2(subdata2)
            loss_value2 = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
            )

            loss_value2.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module2.parameters(), max_grad_norm
            )  # Optional

            optim2.step()
            optim2.zero_grad()

    collector.update_policy_weights_()

    # Logging
    done = tensordict_data.get(("next", "agent_blue", "done"))  # or tensordict_data.get(("next", "agent_red", "done"))
    episode_reward_mean = (
        tensordict_data.get(("next", "agent_blue", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list.append(episode_reward_mean)

    episode_reward_mean2 = (
        tensordict_data.get(("next", "agent_red", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list2.append(episode_reward_mean2)
    pbar.set_description(f"episode_reward_mean_blue/red = {episode_reward_mean},{episode_reward_mean2}", refresh=False)
    pbar.update()

    torch.save(combined_policy[BLUE], "policy_blue.pt")
    torch.save(combined_policy[RED], "policy_red.pt")

# =============training loop finished, now show the learning graph ==========================

plt.plot(episode_reward_mean_list)
plt.xlabel("Training iterations")
plt.ylabel("Reward")
plt.title("Episode reward mean blue")
plt.show()


plt.plot(episode_reward_mean_list2)
plt.xlabel("Training iterations")
plt.ylabel("Reward")
plt.title("Episode reward mean red")
plt.show()














# Optional loading of checkpoint data

# Training Loop ....

# Inference/Rollout