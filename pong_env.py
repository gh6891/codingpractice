import gymnasium as gym
import ale_py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
import cv2
import os


class FrameProcessor:
    def __init__(self):
        self.frame_stack = deque(maxlen=4)
    
    def process(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # 흑백 변환
        state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)  # 크기 조정
        state = np.expand_dims(state, axis=0) / 255.0  # 차원 확장 및 정규화

        self.frame_stack.append(state)  # 프레임 추가

        if len(self.frame_stack) < 4:
            # 4개가 안되면 같은 프레임을 복사하여 채우기
            while len(self.frame_stack) < 4:
                self.frame_stack.append(state)

        return np.array(self.frame_stack)
    
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, action_size)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
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



class PongAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.00025,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        final_epsilon: float = 0.1,
        discount_factor: float = 0.99,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 1000
    ):
        self.env = env
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(env.action_space.n).to(self.device)
        self.target_model = DQN(env.action_space.n).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0  # Target network 업데이트를 위한 카운트

    def get_action(self, state):
        """Epsilon-Greedy 정책으로 행동 선택"""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()  # 탐색
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 차원 추가 후 변환
            state = state.squeeze(2)  # 불필요한 차원 제거 (1, 4, 1, 84, 84) → (1, 4, 84, 84)

            with torch.no_grad():
                q_values = self.model(state)

            return q_values.argmax().item()  # 최적 행동 선택
        
    def store_transition(self, state, action, reward, next_state, done):
        """Replay Memory에 경험 저장"""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """DQN 학습 수행"""
        if len(self.memory) < self.batch_size:
            return  # 데이터 부족할 경우 학습 스킵

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 리스트 → numpy 배열 → PyTorch Tensor 변환
        states = np.array(states)
        next_states = np.array(next_states)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # 불필요한 차원 제거 (Conv2D 입력 문제 해결)
        states = states.squeeze(2)  # (32, 4, 1, 84, 84) → (32, 4, 84, 84)
        next_states = next_states.squeeze(2)

        # 현재 상태의 Q 값 계산
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 다음 상태의 최대 Q 값 계산 (Target Network 사용)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))

        # 손실 계산 및 업데이트
        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon 감소
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

        # Target Network 주기적으로 업데이트
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())


    def train(self, num_episodes=1000, max_timesteps=10000):
        """에이전트 학습"""
        for episode in range(num_episodes):
            state = self.env.reset()
            state = self.preprocess_state(state)  # 이미지 전처리 필요
            total_reward = 0

            for t in range(max_timesteps):
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)

                self.store_transition(state, action, reward, next_state, done)
                self.train_step()

                state = next_state
                total_reward += reward

                if done:
                    break

            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.3f}")

    def save_model(self, filename="pong_dqn.pth"):
        torch.save(self.model.state_dict(), filename)
        print(f"✅ Model saved to {filename}")

    def load_model(self, filename="pong_dqn.pth"):
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename))
            self.model.to(self.device)
            print(f"✅ Model loaded from {filename}")
        else:
            print(f"⚠️ No model file found at {filename}")

    def preprocess_state(self, state):
        """Pong의 원본 이미지를 84x84x4 형태로 변환 (필요한 전처리 과정)"""
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # 흑백 변환
        state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)  # 크기 조정
        state = np.expand_dims(state, axis=0) / 255.0  # 정규화 및 차원 추가
        return state
    
def save_model(self, filename="pong_dqn.pth"):
    """모델 가중치 저장"""
    torch.save(self.model.state_dict(), filename)
    print(f"✅ Model saved to {filename}")

def load_model(self, filename="pong_dqn.pth"):
    """모델 가중치 불러오기"""
    if os.path.exists(filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.to(self.device)
        print(f"✅ Model loaded from {filename}")
    else:
        print(f"⚠️ No model file found at {filename}")
def play():
    """저장된 가중치를 사용하여 Pong 에이전트를 실행"""
    env = gym.make("PongDeterministic-v4", render_mode="rgb_array")  # Pong 환경 생성
    obs, info = env.reset()

    frame_processor = FrameProcessor()  # 프레임 처리기
    obs = frame_processor.process(obs)  # 초기 상태 변환

    agent = PongAgent(env=env)
    agent.load_model("pong_dqn_latest.pth")  # 최근 저장된 가중치 로드

    episode_done = False
    total_reward = 0

    while not episode_done:
        # action = agent.get_action(obs)  # 학습이 아닌 실행이므로 탐험 없이 행동 선택

        obs = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)  # 차원 추가 후 변환
        obs = obs.squeeze(2)  # 불필요한 차원 제거 (1, 4, 1, 84, 84) → (1, 4, 84, 84)

        with torch.no_grad():
            q_values = agent.model(obs)

        action = q_values.argmax().item()  # 최적 행동 선택

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = frame_processor.process(next_obs)  # 다음 상태 전처리
        
        obs = next_obs

        latest_frame = next_obs[-1, 0]
        total_reward += reward
        print("OBS : ", obs.shape)
        # 환경 시각화 (크기 조정 후 출력)
        resized_obs = cv2.resize(latest_frame * 255, (480, 640))  # 정규화 해제 후 시각화
        print(f"resized_obs.shape: {resized_obs.shape}")
        # resized_obs = cv2.cvtColor(resized_obs.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # print(f"resized_obs.shape: {resized_obs.shape}")
        cv2.imshow("Resized Pong", resized_obs.astype(np.uint8))
        cv2.waitKey(10)

        if terminated or truncated:
            episode_done = True  # 종료 상태 확인

    print(f"Total Reward: {total_reward}")
    env.close()

def main():
    """Pong 에이전트를 학습하고 실행"""
    env = gym.make("PongDeterministic-v4", render_mode="rgb_array")  # Pong 환경 생성
    obs, info = env.reset()
    
    frame_processor = FrameProcessor()  # 프레임 처리기
    obs = frame_processor.process(obs)  # 초기 상태 변환

    agent = PongAgent(env=env)
    num_of_episodes = 3000  # 학습할 에피소드 수

    for episode in range(num_of_episodes):
        obs, info = env.reset()
        obs = frame_processor.process(obs)  # 상태 전처리
        episode_done = False
        total_reward = 0

        while not episode_done:
            action = agent.get_action(obs)  # 에이전트의 행동 선택

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = frame_processor.process(next_obs)  # 다음 상태 전처리

            agent.store_transition(obs, action, reward, next_obs, terminated)
            agent.train_step()  # 에이전트 학습

            obs = next_obs
            total_reward += reward

            # # 환경 시각화 (크기 조정 후 출력)
            # resized_obs = cv2.resize(next_obs[0] * 255, (480, 640))  # 정규화 해제 후 시각화
            # resized_obs = cv2.cvtColor(resized_obs.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            # cv2.imshow("Resized Pong", resized_obs.astype(np.uint8))
            # cv2.waitKey(10)

            if terminated or truncated:
                episode_done = True  # 종료 상태 확인

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        agent.save_model("pong_dqn_latest.pth")
    env.close()
    # cv2.destroyAllWindows()  # OpenCV 창 닫기

if __name__ == "__main__":
    # main()
    play()