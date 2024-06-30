import pandas as pd

data = pd.read_csv('train_data.csv', sep='\s*,\s*', engine='python')

X = data['X'].values
Y = data['Y'].values

#初始化
S1 = 0
S2 = 0
S3 = 0
S4 = 0
n = 5

for i in range(n):
    S1 += X[i] * Y[i]
    S2 += X[i]
    S3 += Y[i]
    S4 += X[i] * X[i]

k = (n * S1 - S2 *S3)/(n * S4 - S2 * S2)
b = (S3 - k * S2)/n

print("Coeff:{} Intercept:{}".format(k, b))
#以5个数据为例
#y=1.9x+2.5