"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/5/15 22:42
"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data_X = [
    [0, 0],
    [1, 2],
    [3, 1],
    [8, 8],
    [9, 10],
    [10, 7],
]
# 实例化KMeans, 确定K值
KM = KMeans(n_clusters=2)
# 对数据集data_X进行聚类
result = KM.fit_predict(data_X)
print(result)

# 绘制图像, 确定每个点的x坐标和y坐标, 通过列表生成式拿到数据集的x和y
x = [n[0] for n in data_X]
y = [n[1] for n in data_X]
# 散点图方法: plt.scatter
# 散点图参数: 数据集x坐标, 数据集y坐标, 预测结果(上面得出的result)
# marker类型: marker='o' 表示圆点, marker='*' 表示星点, marker='x' 表示x
plt.scatter(x, y, c=result, marker='*')
# 绘制标题
plt.title("k-means")
# 绘制x坐标和y坐标
plt.xlabel("x坐标")
plt.ylabel("y坐标")
# 绘制图例
plt.legend(["A", "B"])
plt.show()
