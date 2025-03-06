import gymnasium as gym
import numpy as np
from collections import defaultdict
import random

from matplotlib import pyplot as plt

class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.3,
        initial_epsilon: float = 0.99,
        epsilon_decay: float= 0.000001,
        final_epsilon: float = 0.01,
        discount_factor: float = 0.95
    ):
        self.env = env
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.epsilon
        self.training_error = []
        
        self.Q_values = defaultdict(lambda : np.zeros(env.action_space.n))
        self.N_values = defaultdict(int)

    def get_action(self, obs):

        value = random.random()
        if self.epsilon > value:
            action = self.env.action_space.sample()
        else:
            action = self.Q_values[obs].argmax()
        self.epsilon = max(self.final_epsilon, self.epsilon * np.exp(-self.epsilon_decay))
        
        return action, self.epsilon
    
    #model-free monte carlo
    def update(self, episode):
        #Q(S, A) = Q(S, A) + alpha*(G - Q(S, A))
        #G = R + gamma * Q(S, A)
        G = 0
        for obs, action, reward in reversed(episode):
            G = reward + G * self.discount_factor
            self.Q_values[obs][action] = self.Q_values[obs][action] + self.lr * (G - self.Q_values[obs][action])

if __name__ == "__main__":

    env = gym.make('Blackjack-v1', natural=False, sab=False)
    agent = BlackjackAgent(env= env)

    for episode in range(100000):
        episode_terminated = False
        obs, info = env.reset()
        episode_data = []
        print(type(obs))
        while not episode_terminated:
            action, epsilon = agent.get_action(obs)
            next_state, reward, terminated, truncated, info = env.step(action)

            episode_data.append((obs, action, reward))
            obs = next_state
            episode_terminated = terminated or truncated

        agent.update(episode_data)
    
    #state = [4~21, 1~10, 0 or 1]
    # agent.Q_values[obs].argmax()
    fig, axs = plt.subplots()
    
        

        