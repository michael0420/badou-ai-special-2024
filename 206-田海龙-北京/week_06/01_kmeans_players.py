from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 篮球运动员助攻、得分数据集
X = [
    [0.0888, 0.5885],
    [0.1899, 0.8291],
    [0.0747, 0.4974],
    [0.0983, 0.5772],
    [0.1276, 0.5703],
    [0.1671, 0.5835],
    [0.1306, 0.5276],
    [0.1061, 0.5523],
    [0.2446, 0.4007],
    [0.1670, 0.4770],
    [0.2485, 0.4313],
    [0.1227, 0.4909],
    [0.1240, 0.5668],
    [0.1461, 0.5113],
    [0.2315, 0.3788],
    [0.0494, 0.5590],
    [0.1107, 0.4799],
    [0.1121, 0.5735],
    [0.1007, 0.6318],
    [0.2567, 0.4326],
    [0.1956, 0.4280],
    [0.2512, 0.8503],
]

# 进行能力分组，得分好的、助攻好的，得分助攻都好的（MVP）
clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(X)
# 结果和原数组长度一直，相同值代表属于一个分组
print(y_pred)

x = [v[0] for v in X]
y = [v[1] for v in X]

# 散点图展示
plt.scatter(x, y, c=y_pred, marker="x")

#绘制标题
plt.title("Kmeans-Basketball Data")
 
#绘制x轴和y轴坐标
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")
 
#设置右上角图例
# 实际无法显示多个图例 数据marker是一样的
plt.legend(["A","B","C"])

plt.show()
