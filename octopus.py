import random
from collections import defaultdict

class OctopusAgent:
    def __init__(self, gamma=0.9, alpha=0.1, epsilon=0.5):
        self.gamma = gamma  # 할인율
        self.alpha = alpha  # 학습률
        self.epsilon = epsilon  # 탐험 확률
        self.Q = defaultdict(lambda: 0.0)  # 상태-행동 가치 함수 (Q-table)

    def choose_action(self, state):
        """Epsilon-greedy 정책"""
        if random.random() < self.epsilon:
            return "try"  # 탐험
        else:
            return "quit"  # 탐색을 중단 (활용)

    def generate_episode(self, octopus):
        """에피소드 생성 (강화 시도 및 중단)"""
        episode = []
        while not octopus.fail:
            state = (octopus.LV, octopus.count)  # 상태 정의 (LV, 시도 횟수)
            action = self.choose_action(state)
            
            # 행동에 따른 결과 처리
            if action == "quit":
                break  # 중간 종료

            run, reward, count = octopus.try_lvup()  # 강화 시도
            next_state = (octopus.LV, octopus.count)
            
            # 에피소드 기록
            episode.append((state, action, reward))
            
            if run:  # 도망쳤을 때 종료
                break
                
        return episode

    def every_visit_update(self, episode):
        """Every-Visit MC 업데이트"""
        G = 0
        # visited = set()
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward  # Return 계산
            self.Q[(state, action)] += self.alpha * (G - self.Q[(state, action)])  # 업데이트

    def train(self, episodes=1000):
        """에피소드 반복"""
        for episode_num in range(episodes):
            octopus = Octopus()  # 새로운 환경 초기화
            episode = self.generate_episode(octopus) #하나의 에피소드 기록
            self.every_visit_update(episode)
            if (episode_num + 1) % 1000 == 0:
                print(f"에피소드 {episode_num + 1} 완료")





class Octopus:
    def __init__(self):
        self.LV = 2
        self.count = 1
        self.fail = False
        self.reward = 0
        
        self.enhancement_probs = {
            1: {"reward" : 0},
            2: {"success": 0.6, "fail": 0.4, "run": 0.0, "reward": 1},
            3: {"success": 0.5, "fail": 0.5, "run": 0.0, "reward": 3},
            4: {"success": 0.4, "fail": 0.6, "run": 0.0, "reward": 6},
            5: {"success": 0.307, "fail": 0.693, "run": 0.0, "reward": 10},
            6: {"success": 0.205, "fail": 0.765, "run": 0.03, "reward": 15},
            7: {"success": 0.103, "fail": 0.857, "run": 0.04, "reward": 50},
            8: {"success": 0.05, "fail": 0.9, "run": 0.05, "reward": 150},
            9: {"reward": 500}
        }

    def try_lvup(self):
        """강화 시도"""
        if self.fail or self.count > 100 or self.LV == 9 or self.LV == 1:
            self.reward = self.enhancement_probs[self.LV]["reward"]
            self.fail = True
            return True, self.reward, self.count

        outcome = random.choices(
            ["success", "fail", "run"],
            weights=[
                self.enhancement_probs[self.LV].get("success", 0),
                self.enhancement_probs[self.LV].get("fail", 0),
                self.enhancement_probs[self.LV].get("run", 0)
            ]
        )[0]

        if outcome == "fail":
            self.LV = max(self.LV - 1, 2)
        elif outcome == "success":
            self.LV = min(self.LV + 1, 9)
        elif outcome == "run":
            self.fail = True

        self.reward = self.enhancement_probs[self.LV]["reward"]
        self.count += 1

        return self.fail, self.reward, self.count
    
# 학습 실행
mc_agent = OctopusAgent()
mc_agent.train(episodes=1000000000)

# 학습된 Q값 확인
print("학습된 상태-행동 가치 Q(s, a):")
for state_action, value in mc_agent.Q.items():
    print(f"상태: {state_action[0]}, 행동: {state_action[1]}, Q-value: {value:.2f}")