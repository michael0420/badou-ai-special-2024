from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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

clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(X)
print("Kmeans函数",clf)
print("聚类结果",y_pred)

x = [n[0] for n in X]
y = [n[1] for n in X]

plt.scatter(x,y, c=y_pred,marker='o')
plt.title("kmeans-basketball data")
plt.xlabel("assists_per_minute")
plt.ylabel("point_per_minute")
plt.legend(["A","B","C"])
plt.show()