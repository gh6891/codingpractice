import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Gridworld 크기
grid_size = 4
gamma = 1.0  # 할인율 (현재 문제에서는 영향 없음)
theta = 1e-6  # 수렴 기준
timesteplim = 10

timestep = 0
# 초기화: 모든 state의 value를 0으로 설정
V = np.zeros((grid_size, grid_size))

# Terminal states (0,3)과 (3,0)은 항상 0점
terminal_states = [(0, 3), (3, 0)]

# 방향 (상, 하, 좌, 우)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 정책 평가 (반복)
while timestep != timesteplim:
    print(timestep)
    delta = 100  # 값 변화 추적
    new_V = np.copy(V)  # 새로운 V(s) 저장할 배열

    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) in terminal_states:
                continue  # 탈출 지점은 업데이트 X

            reward = -1  # 항상 -1 보상 받음
            new_value = 0

            # 동서남북 4방향 이동 고려
            for action in actions:
                ni, nj = i + action[0], j + action[1]

                # 그리드 바깥으로 나가면 현재 위치 유지
                if ni < 0 or ni >= grid_size or nj < 0 or nj >= grid_size:
                    ni, nj = i, j
                
                new_value += 0.25 * V[ni, nj]  # 0.25 확률로 이동

            # Bellman equation 적용
            new_V[i, j] = reward + new_value

            # 값 변화 추적
            delta = min(delta, abs(new_V[i, j] - V[i, j]))

    V = new_V  # 업데이트
    timestep += 1
    # 수렴 조건 확인
    if delta < theta:
        break

# 결과 출력
print("Final State Values:")
print(np.round(V, 2))
print("sdfsdf")
# ---- 시각화 ---- #
fig, ax = plt.subplots(figsize=(5, 5))

# 그리드 선 그리기
for x in range(5):
    ax.axvline(x, color='black', linewidth=0.5)
for y in range(5):
    ax.axhline(y, color='black', linewidth=0.5)

# Terminal state (0,3) & (3,0) 회색 칠하기
for x, y in terminal_states:
    rect = patches.Rectangle((y, 3 - x), 1, 1, facecolor='gray', edgecolor='black', linewidth=0.5)
    ax.add_patch(rect)

# 각 칸에 가치값 표시
for i in range(grid_size):
    for j in range(grid_size):
        ax.text(j + 0.5, 3.5 - i, f"{V[i, j]:.2f}", va='center', ha='center', fontsize=12, color='black')

ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.set_aspect('equal')

plt.show()
