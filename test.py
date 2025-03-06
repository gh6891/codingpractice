import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
import numpy as np
import gymnasium as gym
import random



# from matplotlib import pyplot as plt

# fig, ax = plt.subplots()
# ax.plot([1,2,3,4], [10, 20, 25, 30])
# ax.set_xlabel("x축 이름")
# ax.set_ylabel("y축 이름")
# ax.set_title("제목")

# plt.show()

data = np.random.rand(10, 10)
print(f"==>> data: {data}")

plt.figure(figsize=(8, 6))
sns.heatmap(data, annot=True, cmap="coolwarm", linewidths= 0.5)

plt.xlabel("X축 (예: Dealer's First Card)")
plt.ylabel("Y축 (예: Player's Sum)")
plt.title("Heatmap 예제")

plt.show()