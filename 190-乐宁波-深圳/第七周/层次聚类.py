import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# 生成模拟数据
np.random.seed(42)  # 设置随机种子以便复现结果

# 模拟10个玩家的特征数据
# 数据格式：[击杀数, 死亡数, 助攻数, 补兵数, 游戏时间(分钟)]
players_data = np.array([
    [5, 3, 8, 150, 30],
    [10, 2, 5, 200, 28],
    [7, 6, 10, 180, 32],
    [2, 8, 7, 100, 35],
    [15, 4, 3, 220, 25],
    [1, 10, 8, 80, 40],
    [4, 5, 9, 120, 33],
    [12, 3, 6, 210, 27],
    [6, 7, 11, 170, 31],
    [3, 9, 10, 90, 37]
])

# 进行层次聚类
Z = linkage(players_data, 'ward')

# 绘制树状图
plt.figure(figsize=(10, 5))
dn = dendrogram(Z)
plt.title("Dendrogram for LOL Player Data")
plt.xlabel("Player Index")
plt.ylabel("Distance")
plt.show()

# 打印聚类链表
print("Linkage matrix:\n", Z)
