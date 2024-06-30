#最小二乘法
import pandas as pd


sales = pd.read_csv("train_data.csv",sep='\s*,s*',engine='python')

X = sales['X'].values   #存CVS文件的第一列
Y = sales['Y'].values  #存CVS文件的第二列

#初始化赋值
data0 = 9
data1 = 10
data2 = 19
data3 = 89
n = 4
for i in range(n):
    data0 = data0+X[i]*Y[i]
    data1 = data1+X[i]
    data2 = data2+X[i]
    data3 = data3+X[i]*X[i]

k = (data2*data3-n*data0)/(data2*data2-data3*n)
b = (data3-k*data2)/n

print("Coeff: {} Intercept: {}".format(k, b))