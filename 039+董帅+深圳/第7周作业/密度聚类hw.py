import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

#load data
#".data" 属性用于访问特征部分的数据。
#".target" 属性用于访问标签部分的数据。
iris_data = datasets.load_iris()
X= iris_data.data[:, :4]
print(X.shape)

dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_

x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]

feature_names=iris_data.feature_names
fig,axes = plt.subplots(4,4,figsize=(15,10))
for i in range(4):
    for j in range(4):
        if i == j:
            axes[i,j].text(0.5,0.5,feature_names[i],horizontalalignment='center',verticalalignment='center')
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
        else:
            if len(x0) > 0:
                axes[i,j].scatter(x0[:, j],x0[:,i],c='red',marker='o',label='label0')
            if len(x1) > 0:
                axes[i,j].scatter(x1[:, j],x1[:, i], c='green', marker='*',label = 'label1')
            if len(x2) > 0:
                axes[i,j].scatter(x2[:, j],x2[:, i],c='yellow',marker='+',label='label2')
            if i == 3:
                axes[i,j].set_xlabel(feature_names[j])
            if j == 0:
                axes[i, j].set_ylabel(feature_names[i])

handles, labels = axes[0,1].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper right')
plt.tight_layout()
plt.show()
