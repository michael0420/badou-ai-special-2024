'''
手推训练的计算过程
'''
import numpy as np

i1 = 0.05
i2 = 0.1

W = [0,0.15,0.2,0.25,0.3,0.4,0.45,0.5,0.55]
Wtemp = W.copy()

b1 = 0.35
b2 = 0.6

to1 = 0.01
to2 = 0.99

k = 0.5

# 正向
zh1 = i1*W[1] + i2*W[2] + b1
ah1 = 1 /(1 + np.exp(-zh1))

zh2 = i1*W[3] + i2*W[4] + b1
ah2 = 1 /(1 + np.exp(-zh2))

zo1 = ah1*W[5] + ah2*W[6] + b2
ao1 = 1 /(1 + np.exp(-zo1))

zo2 = ah1*W[7] + ah2*W[8] + b2
ao2 = 1 /(1 + np.exp(-zo2))
print(zh1,ah1,zh2,ah2)
print(zo1,ao1,zo2,ao2)

# 计算损失函数
Eo1 = 0.5 * (to1 - ao1) ** 2
Eo2 = 0.5 * (to2 - ao2) ** 2
Eto = Eo1 + Eo2
print(Eo1,Eo2,Eto)

# 反向 输出层，链式法则求偏导

Dw5 = (ao1 - to1) * ao1 * (1 - ao1) * ah1
Wtemp[5] = W[5] - k * Dw5
Dw6 = (ao1 - to1) * ao1 * (1 - ao1) * ah2
Wtemp[6] = W[6] - k * Dw6

Dw7 = (ao2 - to2) * ao2 * (1 - ao2) * ah1
Wtemp[7] = W[7] - k * Dw7
Dw8 = (ao2 - to2) * ao2 * (1 - ao2) * ah2
Wtemp[8] = W[8] - k * Dw8


# 反向，隐藏层
Dah1 = (ao1 - to1) * ao1 * (1 - ao1) * W[5] + (ao2 - to2) * ao2 * (1 - ao2) * W[7]
Dah2 = (ao1 - to1) * ao1 * (1 - ao1) * W[6] + (ao2 - to2) * ao2 * (1 - ao2) * W[8]

Dw1 = Dah1 * ah1 * (1 - ah1) * i1
Wtemp[1] = W[1] - k * Dw1

Dw2 = Dah1 * ah1 * (1 - ah1) * i2
Wtemp[2] = W[2] - k * Dw2

Dw3 = Dah2 * ah2 * (1 - ah2) * i1
Wtemp[3] = W[3] - k * Dw3

Dw4 = Dah2 * ah2 * (1 - ah2) * i2
Wtemp[4] = W[4] - k * Dw4


print(Wtemp[1:9])

