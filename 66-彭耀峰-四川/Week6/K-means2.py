'''
根据身高和体重数据将数据聚类为3类，
根据聚类结果可判断出体型偏瘦、正常和偏胖三类
'''

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


X = [[174, 73],
     [170, 80],
     [165, 70],
     [165, 57],
     [165, 55],
     [168, 70],
     [178, 65],
     [170, 90],
     [165, 45],
     [173, 70]
     ]

clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(X)

print(y_pred)

x = [n[0] for n in X]
print(x)
y = [n[1] for n in X]
print(y)

plt.scatter(x, y, c=y_pred, marker='x')
plt.title('Kmeans Data')
plt.xlabel('height')
plt.ylabel('weight')
plt.show()
