import gym
import numpy as np
import copy
import random
from IPython.display import clear_output

env = gym.make("FrozenLake-v0")
q_table = np.zeros([env.observation_space.n, env.action_space.n])

"""Training the agent"""

# Hyperparameters
alpha = 0.9
gamma = 0.9
epsilon = 1
decay_rate = 0.995

# For plotting metrics
all_epochs = []
all_penalties = []
epochs_sum = 0
epochs_sum_list = []
episode_reward = []
epochs_num = 5000
statistics_at = [500, 2000]
for i in range(epochs_num):
    if i in statistics_at:
        print("The " + str(i) + "'th Qtable")
        print(q_table)
    target_Qtable = copy.deepcopy(q_table)
    state = env.reset()
    epochs, penalties, reward, = 0, 0, 0
    done = False
    while (not done) and epochs < 100:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values
        next_state, reward, done, info = env.step(action)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        if done:
            target = reward
            if reward == 0:
                epochs = 99
        else:
            target = reward + gamma * next_max
        new_value = (1 - alpha) * old_value + alpha * target
        target_Qtable[state, action] = new_value

        state = next_state
        epochs += 1
    episode_reward.append(reward)
    epochs_sum += epochs
    if (i+1) % 100 == 0:
        clear_output(wait=True)
        epochs_sum_list.append(epochs_sum/100)
        epochs_sum = 0
    epsilon *= decay_rate
    q_table = copy.deepcopy(target_Qtable)

print("The finish Qtable")
print(q_table)
print(epochs_sum_list)
print(episode_reward)
