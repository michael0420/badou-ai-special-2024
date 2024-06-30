
# 矩阵形式   Y，真实值，列向量 n*1    X 矩阵，n*p  p是特征量。简单的线性y=kx+b，k和b是未知数   B, 未知数构成的列向量 (b k) p*1

# 最后的公式B = (X.TX)^-1X.TY  B = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
# 就能求出k,b的值。 要求会推导最小二乘法公式，涉及到矩阵的求导

import pandas as pd
import numpy as np

sales = pd.read_csv('train_data.csv', sep='\s*,\s*', engine='python')  # 读取CSV

X=sales['X'].values  # [1, 2, 3, 4]
Y=sales['Y'].values  # [6, 5, 7, 10]

# 构建X,Y矩阵
X = np.array([[1, i] for i in X])
Y = np.array(Y)

# 求B
b, k = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)

# 得到线性方程 y=kx+b  1.4x+3.5