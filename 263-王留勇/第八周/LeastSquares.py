"""
最小二乘法
"""

import pandas as pd

sales = pd.read_csv('train_data.csv', sep='\s*,\s*', engine='python')
X = sales['X'].values
Y = sales['Y'].values

# 初始化赋值
s1 = 0
s2 = 0
s3 = 0
s4 = 0
n = 4

for i in range(n):
	s1 += X[i] * Y[i]
	s2 += X[i]
	s3 += Y[i]
	s4 += X[i]**2

# 计算斜率核截距
k = (s2 * s3 - s1 * n) / (s2**2 - s4 * n)
b = (s3 - k * s2) / n
print('k=', k, '\n' 'b=', b)