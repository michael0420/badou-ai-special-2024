
import numpy as np
from sklearn.decomposition import PCA

X = np.array(
    [
        [-1, 2, 66, -1],
        [-2, 6, 58, -1],
        [-3, 8, 45, -2],
        [1, 9, 36, 1],
        [2, 10, 62, 1],
        [3, 5, 83, 2],
    ]
)  # 导入数据，维度为4
X = np.array(
    [
        [10, 15, 29],
        [15, 46, 13],
        [23, 21, 30],
        [11, 9, 35],
        [42, 45, 11],
        [9, 48, 5],
        [11, 21, 14],
        [8, 5, 15],
        [11, 12, 21],
        [21, 20, 25],
    ]
)

k=2

pca = PCA(k)  # 降到2维

# 可以不需要，fit_transport方法已有fit
# pca.fit(X)  # 执行

# 内部其实调用了fit
newX = pca.fit_transform(X)  # 降维后的数据

mean=pca.mean_

print("均值\n",mean)

# 解释方差比率
print("解释方差比率\n",pca.explained_variance_ratio_)

# 与手写PCA结果不同，应该是由于sklearn内部算法的不同，实际使用了SVD（奇异值分解）没有直接用协方差
print("最终结果\n",newX)  # 输出降维后的数据

