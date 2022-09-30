# HW#6 Machine Learning 10-601, Meiirbek Islamov
# Reinforcement Learning using the model-free Q-learning algorithm

# import the necessary libraries
import sys
import numpy as np
from environment import MountainCar

args = sys.argv
assert(len(args) == 9)
mode = str(args[1]) # Mode to run the environment in. Should be either ‘‘raw’’ or ‘‘tile’’
weight_out = args[2] # Path to output the weights of the linear mode
returns_out = args[3] # Path to output the returns of the agent
episodes = int(args[4]) # The number of episodes your program should train the agent for
max_iterations = int(args[5]) # The maximum of the length of an episode
epsilon = float(args[6]) # The value ε for the epsilon-greedy strategy
gamma = float(args[7]) # The discount factor γ
learning_rate = float(args[8]) #The learning rate α of the Q-learning algorithm

# Functions
def step(action):
    state, reward, done = env.step(action)
    return state, reward, done

def initialize_params(state_size, action_size):
    array = np.zeros((state_size, action_size))
    return array

def predict(state, params, intercept, action):
    q_value = []
    for i,item in enumerate(params.T):
        product = 0
        for key, value in state.items():
            product += item[key] * value
            product_int = intercept + product
        q_value.append(product_int)
    return q_value[action]

def get_action(state, params, intercept, epsilon):
    # Actions = [0, 1, 2]
    # q_values = [q(s, 0), q(s, 1), q(s, 2)]
    q_values = []
    for action in range(3):
        q_values.append(predict(state, params, intercept, action))
    optimal_value = max(q_values)
    optimal_action = q_values.index(optimal_value)
    random = np.random.random()
    if random > epsilon:
        return optimal_action
    else:
        return np.random.choice([0, 1, 2])


def update(state, action, params, intercept, learning_rate, gamma, epsilon, mode):
    q_value = predict(state, params, intercept, action)
    new_state, immediate_reward, done = step(action)
    opt_action = get_action(new_state, params, intercept, epsilon)
    target = immediate_reward + gamma * predict(new_state, params, intercept, opt_action)
    diff = q_value - target
    grad_zeros = np.zeros((params.shape))

    if mode == "raw":
        state_mode = np.array((state[0], state[1]))
        grad_zeros[:, action] = state_mode
    else:
        for key, value in state.items():
            grad_zeros[key, action] = 1

    # Update parameters
    params = params - learning_rate * diff * grad_zeros
    intercept = intercept - learning_rate * diff

    return params, intercept, new_state, immediate_reward, done

def train(state, params, intercept, learning_rate, gamma, episodes, max_iterations, epsilon, mode):
    rewards = []
    for i in range(episodes):
        state = env.reset()
        rewards_sum = 0
        for j in range(max_iterations):
            opt_action = get_action(state, params, intercept, epsilon)
            params, intercept, new_state, immediate_reward, done = update(state, opt_action, params, intercept, learning_rate, gamma, epsilon, mode)
            state = new_state
            rewards_sum += immediate_reward
            if done == True:
                break
        rewards.append(rewards_sum)
    return params, intercept, rewards

def write_weights(params, intercept, filename):
    with open(filename, 'w') as f_out:
        f_out.write(str(intercept) + '\n')
        for i, item in enumerate(params):
            for weight in item:
                f_out.write(str(weight) + '\n')

def write_returns(rewards, filename):
    with open(filename, 'w') as f_out:
        for reward in rewards:
            f_out.write(str(reward) + '\n')

# Main body
if mode == "raw":
    state_size = 2 # Raw
else:
    state_size = 2048 # Tile

env = MountainCar(mode)
state = env.reset()

# Resettign the state
state = env.reset()

action_size = 3
intercept = 0
params = initialize_params(state_size, action_size)

params, intercept, rewards_sum = train(state, params, intercept, learning_rate, gamma, episodes, max_iterations, epsilon, mode)
write_weights(params, intercept, weight_out)
write_returns(rewards_sum, returns_out)
