"""
VMAS Football Adversarial Training
==================================

Trains two teams of agents simultaneously, red and blue, within the VMAS Football environment.

The agents can be trained from initial random weights and biases or from existing saved checkpoints.
"""


# Imports
import csv

import torch
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

from matplotlib import pyplot as plt

from tqdm import tqdm

# Module-level constants and configuration
BLUE = 0                                    # Team 0 is the blue team
RED = 1                                     # Team 1 is the red team
LOAD_CHECKPOINT = False                     # Load pre-saved weights and biases?
vmas_device = torch.device("cpu")           # we do not use a GPU

# Configurable Hyperparameters - Baseline parameters v1
frames_per_batch = 39996
total_iterations = 10000
total_frames = frames_per_batch * total_iterations
total_epochs = 4
minibatch_size = 5000
learning_rate_blue = 2e-4
learning_rate_red = 2e-4
value_loss_coef = 0.3                      # importance of critic during policy updates lower for adversarial
max_grad_norm = 0.4                         # PPO parameter ...
clip_epsilon = 0.15                          # clipping parameter
gamma = 0.99                                # discount rate
lmbda = 0.92                                # "lmbda" is the standard spelling in TorchRL (not "lambda")
epsilon = 0.03                              # entropy coefficient
share_parameters_policy = False              # team members share a single policy
set_composite_lp_aggregate(False).set()     # torchRL - disables auto log-probability aggregation - mandatory for PPO
share_parameters_critic = False              # all agents of the same team will use the same critic
mappo = False                                # Set to True for MAPPO, False for IPPO


# Environment Parameters
n_blue_agents = 5
n_red_agents = 5
max_steps = 500                             # limit steps in an episode (truncate) if it doesnt terminate
num_vmas_envs = (
    frames_per_batch // max_steps           # Number of parallel environments that VMAS will run (multi-threading)
)

env = VmasEnv(
    scenario="football",
    num_envs=num_vmas_envs,
    continuous_actions=False,
    max_steps=max_steps,
    device=vmas_device,
    n_blue_agents=n_blue_agents,
    n_red_agents=n_red_agents,
    ai_blue_agents=False,                   # we disable the built-in heuristic agents - all our agents learn
    ai_red_agents=False,
    physically_different=False,             # heterogeneous agents for blue team
    enable_shooting=False                   # kicking/shooting physics added for blue team
)


# Initialisation

torch.manual_seed(0)                        # a seed allows repeatable results for comparative/analytic purposes

# Track and accumulate agent rewards within the environment.
env = TransformedEnv(
    env,
    RewardSum(in_keys=env.reward_keys, out_keys=[("agent_blue", "episode_reward"), ("agent_red", "episode_reward")]),
)

check_env_specs(env)                        # simple self-test to sanity check definitions

# Create either a neural network for every blue agent if share_parameters_policy = True, or a single neural network for
# all blue agents if share_parameters_policy is False.
# We have chosen a depth of 2 and a num_cells of 128 as a compromise between runtime and performance.
policy_net_blue = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agent_blue", "observation"].shape[-1],
        n_agent_outputs=9,
        n_agents=n_blue_agents,
        centralised=False,
        share_params=share_parameters_policy,
        device=vmas_device,
        depth=2,                            # number of fully connected layers
        num_cells=128,
        activation_class=torch.nn.Tanh,
    ),
)

policy_net_red = torch.nn.Sequential(
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

# Wrap the policy network in the standard tensordict format for TorchRL.
policy_module_blue = TensorDictModule(
    policy_net_blue,
    in_keys=[("agent_blue", "observation")],
    out_keys=[("agent_blue", "logits")],
)

policy_module_red = TensorDictModule(
    policy_net_red,
    in_keys=[("agent_red", "observation")],
    out_keys=[("agent_red", "logits")],
)

# Select an action based on the outputs (logits) from the neural network by constructing a distribution to sample from.
# In our case we create a categorical distribution as there are multiple discrete actions. We then sample from this
# distribution using multinomial sampling (selecting each action with probability proportional to the categories weight).
# Sampling from a distribution ensures stochasticity.
policy_blue= ProbabilisticActor(
    module=policy_module_blue,
    # spec=env.action_spec_unbatched,
    in_keys=[("agent_blue", "logits")],
    out_keys=env.action_keys[0],
    distribution_class=Categorical,
    distribution_kwargs={},
    return_log_prob=True,
)

policy_red = ProbabilisticActor(
    module=policy_module_red,
    # spec=env.action_spec_unbatched,
    in_keys=[("agent_red", "logits")],
    out_keys=env.action_keys[1],
    distribution_class=Categorical,
    distribution_kwargs={},
    return_log_prob=True,
)

# Create a single neural network for the critic. We have chosen a larger neural network
# (depth of 2 and a num_cells of 256) for better value approximation.
critic_net_blue = MultiAgentMLP(
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

critic_net_red = MultiAgentMLP(
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

# wrap the critic network in the standard tensordict format for TorchRL
critic_blue = TensorDictModule(
    module=critic_net_blue,
    in_keys=[("agent_blue", "observation")],
    out_keys=[("agent_blue", "state_value")],
)

critic_red = TensorDictModule(
    module=critic_net_red,
    in_keys=[("agent_red", "observation")],
    out_keys=[("agent_red", "state_value")],
)

# Load previous policies for further training if required (checkpointing).
if LOAD_CHECKPOINT:
    policy_blue= torch.load("policy_blue.pt", weights_only= False)
    policy_red = torch.load("policy_red.pt", weights_only= False)

# TensorDictSequential allows us to pass both of our policies sequentially to the collector.
combined_policy = TensorDictSequential(
    policy_blue,
    policy_red,
)

# The collector performs environment rollouts. A synchronised data collector stores a fixed number of frames
# (frames_per_batch), before this batch is used for PPO updates.  This ensures we are learning on policy as
# required by PPO.
collector = SyncDataCollector(
    env,
    combined_policy,
    device=vmas_device,
    storing_device=vmas_device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

# Create a replay buffer to store frames (transitions). By sampling without replacement we are naturally emptying
# the replay buffer every optimisation loop. This ensures we remain on policy by not re-using old frames.
replay_buffer_blue = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=vmas_device
    ),  # We store the frames_per_batch collected at each iteration
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,  # We will sample minibatches of this size
)

replay_buffer_red = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=vmas_device
    ),  # We store the frames_per_batch collected at each iteration
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,  # We will sample minibatches of this size
)

# Calculates both the actor and critic loss for the PPO algorithm. The actor loss uses clipping to ensure stability
# through limited updates.
loss_module_blue = ClipPPOLoss(
    actor_network=combined_policy[BLUE],
    critic_network=critic_blue,
    clip_epsilon=clip_epsilon,
    entropy_coeff=epsilon,
    normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
)

loss_module_red = ClipPPOLoss(
    actor_network=combined_policy[RED],
    critic_network=critic_red,
    clip_epsilon=clip_epsilon,
    entropy_coeff=epsilon,
    normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
)

# Connect our loss module with the tensor dictionary by providing the appropriate keys.
loss_module_blue.set_keys(
    reward=env.reward_keys[0],
    action=env.action_keys[0],
    value=("agent_blue", "state_value"),
    done=("agent_blue", "done"),
    terminated=("agent_blue", "terminated"),
)

loss_module_red.set_keys(
    reward=env.reward_keys[1],
    action=env.action_keys[1],
    value=("agent_red", "state_value"),
    done=("agent_red", "done"),
    terminated=("agent_red", "terminated"),
)

# The GAE computes the Generalised Advantage Estimation which measures actions against the average action in a given
# state.
loss_module_blue.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
)

loss_module_red.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
)

GAE_blue = loss_module_blue.value_estimator
GAE_red = loss_module_red.value_estimator

# Adam stands for Adaptive Moment Estimation.
# The optimiser is used to update the policy during the optimisation loop


optim_blue = torch.optim.Adam(loss_module_blue.parameters(), learning_rate_blue)
optim_red = torch.optim.Adam(loss_module_red.parameters(), learning_rate_red)



# progress bar
pbar = tqdm(total=total_iterations, desc="episode_reward_mean = 0")

# Used to graph training results
episode_reward_mean_list_blue = []
episode_reward_mean_list_red = []

# rollout loop
for tensordict_data in collector:

    # Done and terminated states need their own code due to not conforming to the (S,A,R,S') paradigm.
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

    # Advantage estimation: compute GAE and add it to the tensordict
    with torch.no_grad():
        GAE_blue(
            tensordict_data,                    # must provide full tensordict
            params=loss_module_blue.critic_network_params,
            target_params=loss_module_blue.target_critic_network_params,
        )

    # Ensure only blue agent data goes in the blue replay buffer.
    scrubbed = tensordict_data.exclude("agent_red", ("next", "agent_red"))
    data_view = scrubbed.reshape(-1)
    replay_buffer_blue.extend(data_view)        # this appends "data_view" to the replay buffer

    with torch.no_grad():
        GAE_red(
            tensordict_data,
            params=loss_module_red.critic_network_params,
            target_params=loss_module_red.target_critic_network_params,
        )

    scrubbed = tensordict_data.exclude("agent_blue", ("next", "agent_blue"))
    data_view = scrubbed.reshape(-1)
    replay_buffer_red.extend(data_view)

    # Optimisation loop
    for _ in range(total_epochs):

        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer_blue.sample()
            loss_vals = loss_module_blue(subdata)
            loss_value = (
                    loss_vals["loss_objective"]
                    + value_loss_coef * loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
            )

            # Triggers backpropagation to compute gradients.
            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module_blue.parameters(), max_grad_norm
            )

            # Updates the parameters of policy (weights) based on the gradient values.
            optim_blue.step()

            # Clears the calculated gradients for the next iteration.
            optim_blue.zero_grad()

        for _ in range(frames_per_batch // minibatch_size):
            subdata2 = replay_buffer_red.sample()
            loss_vals = loss_module_red(subdata2)
            loss_value = (
                    loss_vals["loss_objective"]
                    + value_loss_coef * loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module_red.parameters(), max_grad_norm
            )

            optim_red.step()
            optim_red.zero_grad()

    # Update the collector's copies of the policies
    collector.update_policy_weights_()

    # Log episode rewards to produce a training graph
    done = tensordict_data.get(("next", "agent_blue", "done"))  # or tensordict_data.get(("next", "agent_red", "done"))
    episode_reward_mean_blue = (
        tensordict_data.get(("next", "agent_blue", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list_blue.append(episode_reward_mean_blue)

    episode_reward_mean_red = (
        tensordict_data.get(("next", "agent_red", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list_red.append(episode_reward_mean_red)

    # Update the progress bar.
    pbar.set_description(f"episode_reward_mean_blue/red = {episode_reward_mean_blue},{episode_reward_mean_red}", refresh=False)
    pbar.update()

    # Save the new policies for runtime analysis.
    torch.save(combined_policy[BLUE], "policy_blue_clone.pt")
    torch.save(combined_policy[RED], "policy_red_clone.pt")


# Export the training data


with open("results.iter", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "iteration,blue_episode_reward_mean", "red_episode_reward_mean"])
    for i, (blue, red) in enumerate(zip(episode_reward_mean_list_blue, episode_reward_mean_list_red)):
        writer.writerow([i, blue, red])

# Training loop finished, now show the learning graphs.
plt.plot(episode_reward_mean_list_blue)
plt.xlabel("Training iterations")
plt.ylabel("Reward")
plt.title("Episode reward mean blue")
plt.show()

plt.plot(episode_reward_mean_list_red)
plt.xlabel("Training iterations")
plt.ylabel("Reward")
plt.title("Episode reward mean red")
plt.show()
