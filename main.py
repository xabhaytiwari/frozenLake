import numpy as np
import gymnasium as gym
import random
import os
import pickle
# Generate the FrozenLake-v1 environment using 4X4 map and non-slippery version
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="rgb_array")

# Some observations from our environment
print("Observation Space: ", env.observation_space)
print("Sample Observation: ", env.observation_space.sample())
print("Action Space Shape: ", env.action_space)
print("Action Space Sample: ", env.action_space.sample())

state_space = env.observation_space.n
action_space = env.action_space.n

# Create and Initialise the Q-Table
def initialise_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable

Qtable_frozenlake = initialise_q_table(state_space, action_space)

# Define Policies
def greedy_policy(Qtable, state):
    # Only Exploitation
    action = np.argmax(Qtable[state][:])
    return action

def epsilon_greedy_policy(Qtable, state, epsilon):
    random_num = random.uniform(0, 1)

    # If random num > greater than epsilon we exploit
    if random_num > epsilon:
        action = greedy_policy(Qtable, state)
    
    # Explore
    else:
        action = env.action_space.sample()

    return action

# Define Hyperparameters

n_training_episodes = 10000
learning_rate = 0.7

n_eval_episodes = 100

env_id = "FrozenLake-v1"
max_steps = 99 # Max steps per episode
gamma = 0.95
eval_seed = [] 

# Exploration Parameters
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0005 

# Training Loop

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in range(n_training_episodes):
        # Reduce epsilon 
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        # Reset the environment
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)

            # Take action At and observe Rt + 1 and St + 1
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update Q(s,a) := Q(s,a) + lr [R(s,a) + gamma * max Q(s', a') - Q(s, a)]
            Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

            #If terminated or truncated finish the episode
            if terminated or truncated:
                break

            # Our next state is the new state    
            state = new_state 
    return Qtable

Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_frozenlake)

print(Qtable_frozenlake)

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  """
  episode_rewards = []
  for episode in range(n_eval_episodes):
    if seed:
      state, info = env.reset(seed=seed[episode])
    else:
      state, info = env.reset()
    step = 0
    truncated = False
    terminated = False
    total_rewards_ep = 0

    for step in range(max_steps):
      # Take the action (index) that have the maximum expected future reward given that state
      action = greedy_policy(Q, state)
      new_state, reward, terminated, truncated, info = env.step(action)
      total_rewards_ep += reward

      if terminated or truncated:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

# Evaluate our Agent
mean_reward_frozenLake, std_reward_frozenLake = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
print(f"Mean_reward={mean_reward_frozenLake:.2f} +/- {std_reward_frozenLake:.2f}")

# Solve Taxi-v3
env = gym.make("Taxi-v3", render_mode="rgb_array")

state_space = env.observation_space.n
action_space = env.action_space.n

Qtable_taxi = initialise_q_table(state_space, action_space)
print(Qtable_taxi)

# Taxi - v3 Hyperparemeters
# Training parameters
n_training_episodes = 25000   # Total training episodes
learning_rate = 0.7           # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# DO NOT MODIFY EVAL_SEED
eval_seed = [16,54,165,177,191,191,120,80,149,178,48,38,6,125,174,73,50,172,100,148,146,6,25,40,68,148,49,167,9,97,164,176,61,7,54,55,
 161,131,184,51,170,12,120,113,95,126,51,98,36,135,54,82,45,95,89,59,95,124,9,113,58,85,51,134,121,169,105,21,30,11,50,65,12,43,82,145,152,97,106,55,31,85,38,
 112,102,168,123,97,21,83,158,26,80,63,5,81,32,11,28,148] # Evaluation seed, Each seed has a specific starting state

# Environment parameters
env_id = "Taxi-v3"           # Name of the environment
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate

# Exploration parameters
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05           # Minimum exploration probability
decay_rate = 0.005            # Exponential decay rate for exploration prob

Qtable_taxi = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable_taxi)
print(Qtable_taxi)

mean_reward_taxi, std_reward_taxi = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_taxi, eval_seed)
print(f"Mean_reward={mean_reward_taxi:.2f} +/- {std_reward_taxi:.2f}")