from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


data = [[0.0888, 0.5885],
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

#调用KMeans聚合，分3类
kmeans = KMeans(n_clusters=3)

#训练模型 拟合（fit）和预测
y_pred = kmeans.fit_predict(data)

#获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
x = [x[0] for x in data]
y = [y[1] for y in data]
plt.scatter(x, y,c = y_pred,marker='o')

#设置标题
plt.title("Kmeans-Basketball Data")

#设置x轴和y轴
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")

#设置右上角图例
plt.legend(["A","B","C"])
plt.show()


