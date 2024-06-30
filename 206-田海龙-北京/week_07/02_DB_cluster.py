
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn import datasets 
from  sklearn.cluster import DBSCAN

def dbscan(data, eps=0.5, min_samples=5):

    plt.figure(figsize=(18, 6))

    # 绘制数据分布图
    # '''
    plt.subplot(1,2,1)
    plt.title("Original data")
    plt.scatter(data[:, 0], data[:, 1], c="red", marker='o', label='see')  
    plt.xlabel('sepal length')  
    plt.ylabel('sepal width')  
    plt.legend(loc=2)  
    # plt.show()  
    # '''

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)
    label_pred = dbscan.labels_
    
    # 绘制结果
    plt.subplot(1,2,2)
    plt.title("DBSCAN")
    x0 = data[label_pred == 0]
    x1 = data[label_pred == 1]
    x2 = data[label_pred == 2]
    plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')  
    plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')  
    plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')  
    plt.xlabel('sepal length')  
    plt.ylabel('sepal width')  
    plt.legend(loc=2)  

    plt.show() 


iris = datasets.load_iris() 
data = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度
print(data.shape)
dbscan(data, eps=0.4, min_samples=9)