"""
VMAS Football Adversarial Training
==================================

Trains two teams of agents simultaneously, red and blue, within the VMAS Football environment.

The agents can be trained from initial random weights and biases or from existing saved checkpoints.
"""


# Imports

import torch
from sympy.polys.numberfields.galois_resolvents import generate_lambda_lookup

from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

from tensordict.nn import set_composite_lp_aggregate, TensorDictModule, TensorDictSequential

from pathlib import Path

# Module-level constants and configuration
BLUE = 0                                    # Team 0 is the blue team
RED = 1                                     # Team 1 is the red team
LOAD_CHECKPOINT = False                     # Load pre-saved weights and biases?
vmas_device = torch.device("cpu")           # we do not use a GPU

# Inference Parameters
# environment_runs = 1

# Environment Parameters
n_blue_agents = 5
n_red_agents = 5
max_steps = 600                             # limit steps in an episode (truncate) if it doesnt terminate
num_vmas_envs = 1

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

# Let the user select between pre-saved policy checkpoints
def get_policy_file(prompt):
    file_path = None
    while file_path is None:
        policy_file_list = [f.name for f in Path('.').iterdir() if f.is_file() and f.suffix == '.pt']
        for i, f in enumerate(policy_file_list[:20], 1):  # Show first 20
            print(f"{i:2d}. {f}")
        try:
            choice = int(input(prompt))
            file_path = Path(policy_file_list[choice - 1]) if 1 <= choice <= len(policy_file_list) else None
        except:
            file_path = None
    return file_path

# Prompt the user for the number of training runs
def get_environment_runs(prompt):
    value = None
    while value is None:
        try:
            value = int(input(prompt))
        except:
            value = None
    return value

# Initialisation

torch.manual_seed(0)                        # a seed allows repeatable results for comparative/analytic purposes

# Track and accumulate agent rewards within the environment.
env = TransformedEnv(
    env,
    RewardSum(in_keys=env.reward_keys, out_keys=[("agent_blue", "episode_reward"), ("agent_red", "episode_reward")]),
)

check_env_specs(env)                        # simple self-test to sanity check definitions

# obtain the policies and the number of times to run
blue_policy_file =get_policy_file("Select blue policy : ")
red_policy_file = get_policy_file("Select red policy : ")
environment_runs = get_environment_runs("Enter how many environment runs : ")

print()
print("Blue policy file :", blue_policy_ file)
print("Red policy file :", red_policy_file)
print("Number of environment runs :", environment_runs)

#load policies for inference:
policy_blue = torch.load(blue_policy_file, weights_only= False)
policy_red = torch.load(red_policy_file, weights_only= False)

# old code ...
# policy_blue = torch.load("policy_blue.pt", weights_only= False)
# policy_red = torch.load("policy_red.pt", weights_only= False)

# TensorDictSequential allows us to pass both of our policies sequentially to the collector.
combined_policy = TensorDictSequential(
    policy_blue,
    policy_red,
)

combined_policy.eval()

for _ in range (environment_runs):
    with torch.no_grad():                   # torch no grad is a setting for inference that disables gradient calculation.
       env.rollout(
           max_steps=max_steps,
           policy=combined_policy,
           callback=lambda env, _: env.render(),
           auto_cast_to_device=True,
           break_when_any_done=False,
       )


# rollout = env.rollout(max_steps=100)
# rewards = rollout["agent_blue","episode_reward"]  # Shape: (n_agents, max_steps) or (total_steps,)

