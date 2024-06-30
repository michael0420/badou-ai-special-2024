'''
最小二乘法 Least square method
'''
import pandas as pd

# X = [1, 2, 3, 4, 5, 6, 7]
# Y = [4, 6, 7, 10, 14, 15, 18]
data = pd.read_csv('LSM.csv', sep='\s*,\s*', engine='python')  # sep='\s*,\s*'怎么理解， /s是一个tab键
'''
csv逗号分隔，xlsx要另存，不能直接改扩展名
sep：读取csv文件时指定的分隔符，默认为逗号
engine：解析数据的引擎有两种：c、python。默认为 c 解析速度更快，但是特性没有 python 全。如果使用 c 引擎没有的特性时，会自动退化为 python 引擎。
比如使用分隔符进行解析，如果指定分隔符不是单个字符、或者" \s+ "，那么 c 引擎就无法解析了。
'''
X = data['X'].values
Y = data['Y'].values

s1, s2, s3, s4, n = 0, 0, 0, 0, 7
for i in range(n):
    s1 += X[i]
    s2 += Y[i]
    s3 += X[i] * Y[i]
    s4 += X[i] ** 2
k = (n * s3 - s1 * s2) / (n * s4 - s1 ** 2)
b = s2 / n - k * s1 / n
print('最小二乘法拟合公式为y = {}x + {}'.format(k, b))
