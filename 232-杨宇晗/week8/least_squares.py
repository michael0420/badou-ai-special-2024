import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
sales = pd.read_csv('train_data.csv', sep='\s*,\s*', engine='python')
X = sales['X'].values    # 存csv的第一列
Y = sales['Y'].values    # 存csv的第二列

# 初始化赋值
s1 = 0
s2 = 0
s3 = 0
s4 = 0
n = len(X)  # 使用len(X)自动获取数据长度

# 循环累加
for i in range(n):
    s1 += X[i] * Y[i]   # X*Y，求和
    s2 += X[i]          # X的和
    s3 += Y[i]          # Y的和
    s4 += X[i] ** 2     # X的平方，求和

# 计算斜率和截距
k = (n * s1 - s2 * s3) / (n * s4 - s2 ** 2)
b = (s3 - k * s2) / n

# 打印斜率和截距
print("斜率: {:.2f}, 截距: {:.2f}".format(k, b))

# 绘制数据点
plt.scatter(X, Y, color='blue', label='数据点')

# 计算拟合直线的Y值
Y_pred = k * X + b

# 绘制拟合直线
plt.plot(X, Y_pred, color='red', label='拟合直线')

# 设置图例
plt.legend()

# 设置坐标轴标签
plt.xlabel('X')
plt.ylabel('Y')

# 显示图表
plt.show()
