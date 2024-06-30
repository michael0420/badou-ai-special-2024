import pandas as pd
import numpy as np


src_path = 'test.xlsx'
data = pd.read_excel(src_path,engine='openpyxl')

print(data)
print(data.shape)
# 注意 pandas 中不能使用 data[:,0}
X = data['X']
Y = data['Y']

data_numpy = np.array(data)
row = data.iloc[:5]
# 可以直接转为numpy数组
print(data_numpy)
print(data_numpy[:,0])
# 最小二乘法   ∑ (Y-(KX + B))^2 对k和b分别求导，导数等于0
# ∑x*∑y-N∑（x*y） / (∑ x)^2 - N∑x^2
x_y = 0
x = 0
y = 0
x_x =0
n = len(data['X'])
print(n)
for i in range(n):
    x_y = x_y + X[i]*Y[i]
    x = x + X[i]
    y = y + Y[i]
    x_x = x_x +X[i]*X[i]

# 加入了三四个噪音点 ，原本方程是  y = 1.4*x
K = (x*y-n*x_y) / (x**2 - n*x_x)
B =  (y-K*x) / n
print(f'直线的斜率是{K},截距是：{B}')
