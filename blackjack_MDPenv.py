import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


import gymnasium as gym
import numpy as np
from collections import defaultdict
import random

from matplotlib import pyplot as plt

class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.01,
        initial_epsilon: float = 0.99,
        epsilon_decay: float= 0.00001,
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
        #Q(S, A) = Q(S, Q) + alpha * (R + rQ(S', A')-Q(S, A))
        #Q(S, A) = Q(S, A) + alpha*(G - Q(S, A)) MC
        #G = R + gamma * Q(S, A)
        G = 0
        for obs, action, reward in reversed(episode):
            G = reward + G * self.discount_factor

            self.N_values[(obs, action)] += 1
            alpha = 1 / self.N_values[(obs, action)]
            self.Q_values[obs][action] = self.Q_values[obs][action] + alpha * (G - self.Q_values[obs][action])
            # self.Q_values[obs][action] = self.Q_values[obs][action] + self.lr * (G - self.Q_values[obs][action])

if __name__ == "__main__":

    env = gym.make('Blackjack-v1', natural=False, sab=False)
    agent = BlackjackAgent(env= env)

    for episode in range(1000000):
        if episode % 10000 == 0:
            print("episode : ", episode, "epsilon : ", agent.epsilon)
        episode_terminated = False
        obs, info = env.reset()
        episode_data = []
        while not episode_terminated:
            action, epsilon = agent.get_action(obs)
            next_state, reward, terminated, truncated, info = env.step(action)

            episode_data.append((obs, action, reward))
            obs = next_state
            episode_terminated = terminated or truncated
        #MC updates impormation after episode ends
        agent.update(episode_data)
    player_sums = np.arange(4, 22)
    dealer_cards = np.arange(1, 11)

    policy_no_usable_ace = np.zeros((len(player_sums), len(dealer_cards)))
    policy_usable_ace = np.zeros((len(player_sums), len(dealer_cards)))
    
    for i, player_sum in enumerate(player_sums):
        for j, dealer_showing in enumerate(dealer_cards):
            state_no_ace = (player_sum, dealer_showing, 0)
            state_ace = (player_sum, dealer_showing, 1)

            if state_no_ace in agent.Q_values:
                policy_no_usable_ace[i, j] = agent.Q_values[state_no_ace].argmax()
            if state_ace in agent.Q_values:
                policy_usable_ace[i, j] = agent.Q_values[state_ace].argmax()
    
    fig, axs = plt.subplots(1, 2, figsize=(14,6))

    sns.heatmap(policy_no_usable_ace, annot=True, cmap="coolwarm", linewidths=0.5,
            xticklabels=dealer_cards, yticklabels=player_sums, ax=axs[0])
    axs[0].set_title("Policy (No Usable Ace)")
    axs[0].set_xlabel("Dealer's First Card")
    axs[0].set_ylabel("Player's Sum")

    # 🃏 두 번째 히트맵 (Usable Ace)
    sns.heatmap(policy_usable_ace, annot=True, cmap="coolwarm", linewidths=0.5,
                xticklabels=dealer_cards, yticklabels=player_sums, ax=axs[1])
    axs[1].set_title("Policy (Usable Ace)")
    axs[1].set_xlabel("Dealer's First Card")
    axs[1].set_ylabel("Player's Sum")

    # 전체 그래프 조정
    plt.tight_layout()  # 서브플롯 간 여백 조정
    plt.show()
    #state = [4~21, 1~10, 0 or 1]
    # agent.Q_values[obs].argmax()
    # fig, axs = plt.subplots()
    
        

        