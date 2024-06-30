import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN
# 导入数据集，并提取数据集的前四维作为待处理项
iris= datasets.load_iris()
X=iris.data[:,:4]
# 设置密度聚类的半径为0.4，最小数为9
dbscan=DBSCAN(eps=0.4,min_samples=9)
dbscan.fit(X)
label_pred=dbscan.labels_
# 将分出的三类结果分别取出，方便画图
x0=X[label_pred==0]
x1=X[label_pred==1]
x2=X[label_pred==2]
# 画图
plt.figure()
plt.scatter(x0[:,0],x0[:,1],c='r',s=25,marker='*',label='label0')
plt.scatter(x1[:,0],x1[:,1],c='b',s=25,marker='^',label='label1')
plt.scatter(x2[:,0],x2[:,1],c='g',s=25,marker='.',label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()



