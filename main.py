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
mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")