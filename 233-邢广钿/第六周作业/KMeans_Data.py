from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# KMeans计算的数据集 第一列表示助攻数 第二列表示得分数
DATA = [[0.0888, 0.5885],
     [0.1399, 0.8291],
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
     [0.1956, 0.4280]
    ]
#定义Kmeans的簇数是3  即类聚为3大类
km = KMeans(n_clusters=3)
#载入数据得到结果
y_pred = km.fit_predict(DATA)
print("y_pred = ",y_pred)

#将数据可视化

#获取数据集的第一列和第二列数据 第一列为X轴，第二列为Y轴
x = [n[0] for n in DATA]
y = [n[1] for n in DATA]

#绘制二维坐标图 x:x轴数据 y:y轴数据 c=y_pred聚类预测结果 marker:绘制出来的点用'x"表示
plt.scatter(x, y, c=y_pred, marker='x')

# 绘制标题
plt.title("Kmeans")
# 绘制x轴和y轴坐标
plt.xlabel("zhugong")
plt.ylabel("defen")
# 设置右上角图例
plt.legend(["A", "B", "C"])

# 显示图形
plt.show()
