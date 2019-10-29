import gym
import random
import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from time import sleep

env = gym.make('CartPole-v1')
states = env.observation_space.shape[0]
actions = env.action_space.n

def agent(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model = agent(env.observation_space.shape[0], env.action_space.n)

from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy
sarsa = SARSAAgent(model = model, policy = EpsGreedyQPolicy(), nb_actions = env.action_space.n)
# sarsa.compile('adam', metrics = ['mse'])
# sarsa.fit(env, nb_steps = 50000, visualize = False, verbose = 1)
# scores = sarsa.test(env, nb_episodes = 100, visualize= True)

# sarsa.save_weights('1-sarsa_weights.h5f', overwrite=True)
sarsa.load_weights('1-sarsa_weights.h5f')
_ = sarsa.test(env, nb_episodes = 100, visualize= True)
print('Average score over 100 test games:{}'.format(np.mean(_.history['episode_reward'])))