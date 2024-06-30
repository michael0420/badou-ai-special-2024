# 1.该方法是用来描述数学模型与真实数据点之间的关系
# 2.通过计算两者之间的最小残差平方和来确定最佳的数学模型(求模型的k和b)
import pandas as pd
import numpy as np

data = pd.read_csv("train_data.csv",sep='\s*,\s*', engine='python')

X = np.array(data['X'])
Y = np.array(data['Y'])
S1 = 0
S2 = 0
S3 = 0
S4 = 0
n = 4

for i in range(4):
    S1 = S1 + X[i]*Y[i]
    S2 = S2 + X[i]
    S3 = S3 + Y[i]
    S4 = S4 + X[i]*X[i]

k = (n*S1 - S2*S3) / (n*S4 - S2*S2)
b = (S3 - k*S2) / n
# k = (S2*S3-n*S1)/(S2*S2-S4*n)
# b = (S3 - k*S2)/n
print(k)
print(b)











