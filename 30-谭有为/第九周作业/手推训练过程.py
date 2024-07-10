import numpy as  np
import math

#激活函数sigmoid
def sigmoid(x):
     return 1 / (1 + math.exp(-x))

#激活函数的导数
def  sigmoid_derivatives(x):
    fx=sigmoid(x)
    return fx * (1 - fx)

i1=0.05
i2=0.10
w1=0.15
w2=0.2
w3=0.25
w4=0.3
w5=0.4
w6=0.45
w7=0.5
w8=0.55
b1=0.35
b2=0.6
target1=0.01
target2=0.99

#正向传播

#隐藏层
h1=i1*w1+i2*w2+b1
h2=i1*w3+i2*w4+b1
print(h1,h2)
ah1=sigmoid(h1)
ah2=sigmoid(h2)
print(ah1,ah2)

#输出层
zo1=ah1*w5+ah2*w6+b2
zo2=ah1*w7+ah2*w8+b2
print(zo1,zo2)
ao1=sigmoid(zo1)
ao2=sigmoid(zo2)
print(ao1,ao2)

#计算损失函数
Eo1=(ao1-target1)**2/2
Eo2=(ao2-target2)**2/2
E_total=Eo1+Eo2
print(E_total)

#反向传播
d_E_total_outo1=-(target1-ao1)
d_outo1_neto1=sigmoid_derivatives(zo1)
d_neto1_w5=ah1
d_net01_w6=ah2
print(d_E_total_outo1,d_outo1_neto1,d_neto1_w5)


d_E_total_outo2=-(target2-ao2)
d_outo2_neto2=sigmoid_derivatives(zo2)
d_neto1_w7=ah1
d_neto1_w8=ah2

E_total_w5=d_E_total_outo1*d_outo1_neto1*d_neto1_w5
E_total_w6=d_E_total_outo1*d_outo1_neto1*d_net01_w6
E_total_w7=d_E_total_outo2*d_outo2_neto2*d_neto1_w7
E_total_w8=d_E_total_outo2*d_outo2_neto2*d_neto1_w8
print(E_total_w5,E_total_w6,E_total_w7,E_total_w8)

#学习率
lr=0.5

#输出层权值更新
w5_new=w5-lr*E_total_w5
w6_new=w6-lr*E_total_w6
w7_new=w7-lr*E_total_w7
w8_new=w8-lr*E_total_w8
print('隐藏层的新权重值为：',w5_new,w6_new,w7_new,w8_new)

d_Eo1_outo1=-(target1-ao1)
d_Eo2_outo2=-(target2-ao2)
d_neto1_outh1=w5
d_neto1_outh2=w6
d_neto2_outh1=w7
d_neto2_outh2=w8
d_Eo1_outh1=d_Eo1_outo1*d_outo1_neto1*d_neto1_outh1
d_Eo1_outh2=d_Eo1_outo1*d_outo1_neto1*d_neto1_outh2
d_Eo2_outh1=d_Eo2_outo2*d_outo2_neto2*d_neto2_outh1
d_Eo2_outh2=d_Eo2_outo2*d_outo2_neto2*d_neto2_outh2

d_Eo2_outo1=-(target2-ao2)
d_neto2_outh1=w7
d_Eo2_outh1=d_Eo2_outo1*d_outo2_neto2*d_neto2_outh1

d_E_total_outh1=d_Eo1_outh1+d_Eo2_outh1
d_E_total_outh2=d_Eo1_outh2+d_Eo2_outh2
d_outh1_neth1=sigmoid_derivatives(h1)
d_outh2_neth2=sigmoid_derivatives(h2)
d_neth1_w1=i1
d_neth1_w2=i2
d_neth1_w3=i1
d_neth1_w4=i2

E_total_w1=d_E_total_outh1*d_outh1_neth1*d_neth1_w1
E_total_w2=d_E_total_outh1*d_outh1_neth1*d_neth1_w2
E_total_w3=d_E_total_outh2*d_outh2_neth2*d_neth1_w3
E_total_w4=d_E_total_outh2*d_outh2_neth2*d_neth1_w4
#print(d_E_total_outh1,d_outh1_neth1,d_neth1_w1)
print(E_total_w1,E_total_w2,E_total_w3,E_total_w4)

#更新输入层权重
w1_new=w1-lr*E_total_w1
w2_new=w2-lr*E_total_w2
w3_new=w3-lr*E_total_w3
w4_new=w4-lr*E_total_w4
print('输入层的新权重值为：',w1_new,w2_new,w3_new,w4_new)

#第二次正向传播
h1_new=i1*w1_new+i2*w2_new+b1
h2_new=i1*w3_new+i2*w4_new+b1
ah1_new=sigmoid(h1_new)
ah2_new=sigmoid(h2_new)

zo1_new=ah1_new*w5_new+ah2_new*w6_new+b2
zo2_new=ah1_new*w7_new+ah2_new*w8_new+b2
ao1_new=sigmoid(zo1_new)
ao2_new=sigmoid(zo2_new)

Eo1_new=(ao1_new-target1)**2/2
Eo2_new=(ao2_new-target2)**2/2
E_total_new=Eo1_new+Eo2_new
print(E_total_new)
print('一次迭代后，总误差的差值为：',E_total-E_total_new)



