import numpy as np

w1 = .15
w2 = .05
w3 = .25
w4 = .30
w5 = .40
w6 = .45
w7 = .50
w8 = .55
b1 = .35
b2 = .60

i1 = 0.05
i2 = 0.10

o1 = 0.01
o2 = 0.99

learn_rate = 0.5


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mean_squared_error(y_ture, y_pred):
    import numpy as np
    y_ture = np.array(y_ture)
    y_pred = np.array(y_pred)

    squared_difference = np.square(y_ture - y_pred)
    mse = np.mean(squared_difference)
    return mse


# 前向传播
zh1 = w1 * i1 + w2 * i2 + b1
zh2 = w3 * i1 + w4 * i2 + b1
ah1 = sigmoid(zh1)
ah2 = sigmoid(zh2)
print('ah1, ah2:', ah1, ah2)

zo1 = w5 * ah1 + w6 * ah2 + b2
zo2 = w7 * ah1 + w8 * ah2 + b2
ao1 = sigmoid(zo1)
ao2 = sigmoid(zo2)
print('ao1, ao2:', ao1, ao2)

mse = mean_squared_error([ao1, ao2], [o1, o2])
print('mse:', mse)

# 反向传播开始
# 计算输出层的梯度
delta_o1 = (ao1 - o1) * ao1 * (1 - ao1)
delta_o2 = (ao2 - o2) * ao2 * (1 - ao2)
print('delta_o1, delta_o2:', delta_o1, delta_o2)

# 计算隐藏层的梯度
delta_h1 = delta_o1 * w5 + delta_o2 * w7
delta_h2 = delta_o1 * w6 + delta_o2 * w8
delta_h1 *= ah1 * (1 - ah1)
delta_h2 *= ah2 * (1 - ah2)

# 更新权重和偏置
w5 -= learn_rate * delta_o1 * ah1
w6 -= learn_rate * delta_o1 * ah2
w7 -= learn_rate * delta_o2 * ah1
w8 -= learn_rate * delta_o2 * ah2
b2 -= learn_rate * (delta_o1 + delta_o2)

w1 -= learn_rate * delta_h1 * i1
w2 -= learn_rate * delta_h1 * i2
w3 -= learn_rate * delta_h2 * i1
w4 -= learn_rate * delta_h2 * i2
b1 -= learn_rate * (delta_h1 + delta_h2)

# 打印更新后的权重和偏置（可选）
print("Updated Weights and Biases:")
print("w1:", w1, "w2:", w2, "w3:", w3, "w4:", w4)
print("w5:", w5, "w6:", w6, "w7:", w7, "w8:", w8)
print("b1:", b1, "b2:", b2)
