#原始数据输入
"""
第一部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据
总共20行，每行两列数据
第一列表示球员每分钟助攻数：assists_per_minute
第二列表示球员每分钟得分数：points_per_minute
"""
X = [[0.0888, 0.5885],
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

from sklearn.cluster import KMeans
clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(X)

print(clf)
print("y_pred = ",y_pred)
import numpy as np
import matplotlib.pyplot as plt

x = [n[0] for n in X]
# print(x)
y = [n[1] for n in X]
# print(y)

cmap = plt.cm.get_cmap('viridis',3)
colors = [cmap(0), cmap(1), cmap(2)]
# colors = [cmap(i) for i in np.linspace(0,1,3)]
plt.scatter(x,y,c=[colors[i] for i in y_pred],marker='*')
proxies = [plt.Line2D([0],[0],marker='*',
                      linestyle='none',color=c) for c in colors]
plt.legend(proxies,['Cluster{}'.format(i+1)for i in range(3)],
           loc='upper right')
plt.title('Kmeans-Basketball Data')
plt.xlabel('assists')
plt.ylabel('points')

plt.show()