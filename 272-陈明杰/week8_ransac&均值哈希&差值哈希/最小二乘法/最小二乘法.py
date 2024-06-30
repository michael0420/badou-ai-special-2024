import pandas as pd

# 读取文件
sales = pd.read_csv("train_data.csv", sep='\s*,\s*', engine='python')
print(sales)

# 把文件转变成一维数组
X = sales['X'].values  # 存csv的第一列
print(X)
Y = sales['Y'].values  # 存csv的第二列
print(Y)

s1 = 0
s2 = 0
s3 = 0
s4 = 0
n = 4

# 循环累加
for i in range(n):
    s1 = s1 + X[i] * Y[i]  # X*Y，求和
    s2 = s2 + X[i]  # X的和
    s3 = s3 + X[i] * X[i]  # X**2，求和
    s4 = s4 + Y[i]  # Y的和

# 计算斜率和截距,最小二乘法的公式
k = ((n * s1) - (s2 * s4)) / ((n * s3) - s2 * s2)
b = (s4 - k * s2) / n

print(k)
print(b)
