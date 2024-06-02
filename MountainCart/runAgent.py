import pickle
import sys
import gym
import numpy as np

# MAX_EPISODE_STEPS = 1000
# MAX_EPISODE_STEPS = 500
MAX_EPISODE_STEPS = 210
# env = gym.make("MountainCar-v0")#, render_mode="human")#,max_episode_steps=10)
env = gym.make("MountainCar-v0", render_mode="human")#,max_episode_steps=10)
# env = gym.wrappers.TimeLimit(env, max_episode_steps=30)

LEARNING_RATE = 0.1
# LEARNING_RATE = 0.5
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 1

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Initialize Q table
# q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
with open('q_table.dat','rb') as f: q_table = pickle.load(f)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))  # We use this tuple to look up the Q values for the available actions in the Q-table

def dumpTable():
    with open('q_table.dat', 'wb') as f:
        pickle.dump(q_table, f)

for episode in range(EPISODES):
    initial_state, _ = env.reset()
    discrete_state = get_discrete_state(initial_state)
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(f"Episode: {episode}")
    else:
        render = False

    step = 0
    while not done and step<MAX_EPISODE_STEPS:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if episode % SHOW_EVERY == 0:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
            print(f"Reached goal in episode {episode}")

        discrete_state = new_discrete_state
        step += 1

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()

# Save the Q-table after training
with open('q_table.dat', 'wb') as f:
    pickle.dump(q_table, f)


