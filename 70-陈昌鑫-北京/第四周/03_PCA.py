import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def PCA(src_arr, dimension):
    des_arr = src_arr
    des_arr = des_arr - des_arr.mean(axis=0)
    covariance_matrix = np.dot(des_arr.T, des_arr) / des_arr.shape[0]
    print(des_arr.shape[0])
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    index_arr = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:,index_arr[:dimension]]
    des_arr = np.dot(des_arr, eigenvectors)
    return des_arr

src_arr, labels=load_iris(return_X_y=True) #加载数据
des_arr = PCA(src_arr, 2)
red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
for i in range(len(des_arr)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
    if labels[i]==0:
        red_x.append(des_arr[i][0])
        red_y.append(des_arr[i][1])
    elif labels[i]==1:
        blue_x.append(des_arr[i][0])
        blue_y.append(des_arr[i][1])
    else:
        green_x.append(des_arr[i][0])
        green_y.append(des_arr[i][1])
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()
