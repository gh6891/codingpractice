#states : Two locations, maximun of 20 cars at each
#Actions : Move up to 5 cars between locations overnight
#Reward : $10 for each car rental (must be available)
#Transitions : cars returned and requested randomly
#Poisson distribution, n returns/ requests 
#1st location : average requests = 3, average returns = 3
#2nd location : average requests = 4, average returns = 2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
# 환경 설정
max_cars = 20  # 각 지점 최대 차량 수
max_move = 5    # 최대 이동 가능 차량 수
gamma = 0.9     # 할인율
theta = 1e-4    # 수렴 기준
rental_reward = 10  # 차량 대여 시 얻는 보상

# Poisson 확률 계산 (미리 저장하여 속도 최적화)
poisson_cache = {}
def poisson_prob(n, lam):
    """포아송 분포 확률 P(X=n) 계산 (캐시 활용)"""
    key = (n, lam)
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]

# 가치 함수 초기화
V = np.zeros((max_cars + 1, max_cars + 1))

# 정책 초기화 (모든 상태에서 0대 이동)
policy = np.zeros((max_cars + 1, max_cars + 1), dtype=int)
# 정책 평가 & 개선 반복
while True:
    print("11")
    # 1. 정책 평가 (Policy Evaluation)
    while True:
        print("정책평가 시작")
        delta = 0
        new_V = np.copy(V)
        
        for s1 in range(max_cars + 1):  # 1번 지점 차량 개수 (y축)
            for s2 in range(max_cars + 1):  # 2번 지점 차량 개수 (x축)
                action = policy[s1, s2]  # 현재 정책에서의 행동
                new_s1 = min(s1 - action, max_cars)  # 1번 지점 차량 상태 갱신
                new_s2 = min(s2 + action, max_cars)  # 2번 지점 차량 상태 갱신
                
                # 현재 가치 계산
                value = 0
                for req1 in range(11):  # 최대 10대 요청 고려 (1번 지점)
                    for req2 in range(11):  # 최대 10대 요청 고려 (2번 지점)
                        prob1 = poisson_prob(req1, 3)  # 1번 지점 요청 확률
                        prob2 = poisson_prob(req2, 4)  # 2번 지점 요청 확률
                        prob = prob1 * prob2  # 독립 확률

                        rent1 = min(new_s1, req1)  # 1번 지점에서 실제 렌트된 차량 수
                        rent2 = min(new_s2, req2)  # 2번 지점에서 실제 렌트된 차량 수
                        reward = (rent1 + rent2) * rental_reward  # 보상

                        # 차량 반납 (Poisson)
                        for ret1 in range(11):
                            for ret2 in range(11):
                                prob_ret1 = poisson_prob(ret1, 3)
                                prob_ret2 = poisson_prob(ret2, 2)
                                prob_ret = prob_ret1 * prob_ret2

                                final_s1 = min(new_s1 - rent1 + ret1, max_cars)
                                final_s2 = min(new_s2 - rent2 + ret2, max_cars)

                                value += prob * prob_ret * (reward + gamma * V[final_s1, final_s2])
                
                new_V[s1, s2] = value
                delta = max(delta, abs(V[s1, s2] - new_V[s1, s2]))
                print(f"==>> delta: {delta}")

        V = new_V
        if delta < theta:
            break
    print("정책개선 시작")
    # 2. 정책 개선 (Policy Improvement)
    policy_stable = True
    for s1 in range(max_cars + 1):
        for s2 in range(max_cars + 1):
            old_action = policy[s1, s2]
            best_action = None
            best_value = float('-inf')

            for action in range(-max_move, max_move + 1):  # 차량 이동 (-5 ~ +5)
                if (0 <= s1 - action <= max_cars) and (0 <= s2 + action <= max_cars):  
                    new_s1 = min(s1 - action, max_cars)
                    new_s2 = min(s2 + action, max_cars)

                    value = 0
                    for req1 in range(11):
                        for req2 in range(11):
                            prob1 = poisson_prob(req1, 3)
                            prob2 = poisson_prob(req2, 4)
                            prob = prob1 * prob2

                            rent1 = min(new_s1, req1)
                            rent2 = min(new_s2, req2)
                            reward = (rent1 + rent2) * rental_reward

                            for ret1 in range(11):
                                for ret2 in range(11):
                                    prob_ret1 = poisson_prob(ret1, 3)
                                    prob_ret2 = poisson_prob(ret2, 2)
                                    prob_ret = prob_ret1 * prob_ret2

                                    final_s1 = min(new_s1 - rent1 + ret1, max_cars)
                                    final_s2 = min(new_s2 - rent2 + ret2, max_cars)

                                    value += prob * prob_ret * (reward + gamma * V[final_s1, final_s2])

                    if value > best_value:
                        best_value = value
                        best_action = action

            policy[s1, s2] = best_action
            if old_action != best_action:
                policy_stable = False

    if policy_stable:
        break

# ---- 결과 시각화 ---- #
plt.figure(figsize=(8, 6))
sns.heatmap(policy, cmap="coolwarm", annot=True, fmt=".0f", xticklabels=range(21), yticklabels=range(21))
plt.xlabel("2번 장소 차량 수")
plt.ylabel("1번 장소 차량 수")
plt.title("최적 정책 (Optimal Policy)")
plt.show()
