import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
first_two_cols = iris.data[:, :2]
print(first_two_cols)
# sales=pd.read_csv('train_data.csv',sep='\s*,\s*',engine='python')  #读取CSV
X=first_two_cols[:,0]    #存csv的第一列
Y=first_two_cols[:,1]    #存csv的第二列

#初始化赋值
s1 = 0
s2 = 0
s3 = 0
s4 = 0
n = 4       ####你需要根据的数据量进行修改

#循环累加
for i in range(n):
    s1 = s1 + X[i]*Y[i]     #X*Y，求和
    s2 = s2 + X[i]          #X的和
    s3 = s3 + Y[i]          #Y的和
    s4 = s4 + X[i]*X[i]     #X**2，求和

#计算斜率和截距
k = (s2*s3-n*s1)/(s2*s2-s4*n)
b = (s3 - k*s2)/n
print("Coeff: {} Intercept: {}".format(k, b))
# 计算直线上的两个点，以便绘制直线
x_min, x_max = X.min() - 1, X.max() + 1
y_line = k * np.array([x_min, x_max]) + b


# 绘制散点图
plt.scatter(X,Y,  c="yellow",  edgecolor='k')
plt.xlabel('sepal length)')
plt.ylabel('sepal width)')
plt.title('first two features fit the plot')

# 添加直线到图中
plt.plot(np.array([x_min, x_max]), y_line, color='red', label=f'y = {k}x + {b}')

# 添加图例
plt.legend()

# 显示图形
plt.show()