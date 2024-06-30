from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

x = load_iris()
# print(x)

# 读取数据值
x_data = x.data
x_target = x.target

# 中心化
x_centr = x_data - x_data.mean()
# print(x_centr)

# 求协方差矩阵
x_cov = np.dot(x_centr.T, x_centr)/x_centr.shape[0]

# print(x_cov)

# 求特征值和特征向量
a, b =  np.linalg.eig(x_cov)
# print(a)
# print(b)

# 对特征值按从大到小排列并记录其索引
a_index = np.argsort(-a)
# print(a_index)

# 降维矩阵到2，取前两列特征向量
a_W = b[:,a_index[0:2]]
# print(a_W)

# 对原始数据降维
a_new = np.dot(x_centr, a_W)
print(a_new.shape)

#查看数据分布
R_x,R_y=[],[]
G_x,G_y=[],[]
B_x,B_y=[],[]

for i in range(len(a_new)):
    if x_target[i] == 0:
        R_x.append(a_new[i,0])
        R_y.append(a_new[i,1])

    elif x_target[i] == 1:
        G_x.append(a_new[i,0])
        G_y.append(a_new[i,1])

    else:
        B_x.append(a_new[i,0])
        B_y.append(a_new[i,1])

plt.scatter (R_x
              ,R_y
              # ,c= 'red'
              , marker='o')
plt.scatter (G_x
              ,G_y
              ,marker ='+'
              ,c = 'green')
plt.scatter (B_x
              ,B_y
              ,marker ='8'
              , c = 'blue')
plt.show()