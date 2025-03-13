import gymnasium as gym
import ale_py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import cv2

class FrameProcessor:
    def __init__(self):
        pass
    
    def process(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # 흑백 변환
        state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)  # 크기 조정
        return np.expand_dims(state, axis=0) / 255.0

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def __len__(self):
        return len(self.buffer)



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



gym.register_envs(ale_py)

env = gym.make("PongDeterministic-v4", render_mode="rgb_array")
obs, info = env.reset()
print(obs.shape)


for _ in range(300):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    print(obs.shape)
    resized_obs = cv2.resize(obs, (480, 640))  # (160 -> 80, 210 -> 105)

    cv2.imshow("Resized Pong", resized_obs)  # 작은 창으로 출력
    cv2.waitKey(10)

    if terminated or truncated:
        obs, info = env.reset()

env.close()