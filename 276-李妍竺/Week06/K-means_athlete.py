# 运动员实例
#调用K-means
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

X= np.array([[0.0888, 0.5885],
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
     ])

print(X)

clf = KMeans(n_clusters=3)  # 分成3类，clf:分类器
y_pred = clf.fit_predict(X)  #载入数据集X，并且将聚类的结果赋值给y_pred

print(clf)
print('y_pred',y_pred)


#绘图
# 需将X用np进行转化后，才可用转置
x = X.T[0]
y = X.T[1]

print(x)
print(y)

'''

x = [n[0] for n in X]   #第n行数据的0位
y = [n[1] for n in X]
print(x)
print(y)
'''


#散点图：参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;

plt.scatter(x,y,c=y_pred,marker='o')
plt.title('K-means_basketball_data')
plt.xlabel('assists_per_minute')
plt.ylabel('points_per_minute')

plt.legend('A')  #添加示例

plt.show()

