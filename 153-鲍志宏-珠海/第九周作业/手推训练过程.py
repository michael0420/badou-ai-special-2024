import numpy as np

# Sigmoid 函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 输入数据和期望输出
inputs = np.array([0.05, 0.10])
expected_output = np.array([0.01, 0.99])

# 初始化权重和偏置
w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55
b1, b2 = 0.35, 0.60

# 学习率
learning_rate = 0.5

# 前向传播
def forward_propagation(inputs, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2):
    z_h1 = w1 * inputs[0] + w2 * inputs[1] + b1
    z_h2 = w3 * inputs[0] + w4 * inputs[1] + b1
    a_h1 = sigmoid(z_h1)
    a_h2 = sigmoid(z_h2)
    z_o1 = w5 * a_h1 + w6 * a_h2 + b2
    z_o2 = w7 * a_h1 + w8 * a_h2 + b2
    a_o1 = sigmoid(z_o1)
    a_o2 = sigmoid(z_o2)
    return z_h1, z_h2, a_h1, a_h2, z_o1, z_o2, a_o1, a_o2

# 计算初始输出和误差
z_h1, z_h2, a_h1, a_h2, z_o1, z_o2, a_o1, a_o2 = forward_propagation(inputs, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2)
E_o1 = 0.5 * (expected_output[0] - a_o1) ** 2
E_o2 = 0.5 * (expected_output[1] - a_o2) ** 2
E_total = E_o1 + E_o2

print(f'初始总误差: {E_total}')

# 打印初始前向传播结果
print(f'z_h1: {z_h1}, a_h1: {a_h1}')
print(f'z_h2: {z_h2}, a_h2: {a_h2}')
print(f'z_o1: {z_o1}, a_o1: {a_o1}')
print(f'z_o2: {z_o2}, a_o2: {a_o2}')

# 反向传播
for epoch in range(10000):  # 训练10000次迭代
    # 前向传播
    z_h1, z_h2, a_h1, a_h2, z_o1, z_o2, a_o1, a_o2 = forward_propagation(inputs, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2)

    # 计算输出层的误差导数
    delta_o1 = (a_o1 - expected_output[0]) * sigmoid_derivative(a_o1)
    delta_o2 = (a_o2 - expected_output[1]) * sigmoid_derivative(a_o2)

    # 计算隐藏层的误差导数
    delta_h1 = (delta_o1 * w5 + delta_o2 * w7) * sigmoid_derivative(a_h1)
    delta_h2 = (delta_o1 * w6 + delta_o2 * w8) * sigmoid_derivative(a_h2)

    # 更新输出层到隐藏层的权重和偏置
    w5 -= learning_rate * delta_o1 * a_h1
    w6 -= learning_rate * delta_o1 * a_h2
    w7 -= learning_rate * delta_o2 * a_h1
    w8 -= learning_rate * delta_o2 * a_h2

    b2 -= learning_rate * (delta_o1 + delta_o2)

    # 更新隐藏层到输入层的权重和偏置
    w1 -= learning_rate * delta_h1 * inputs[0]
    w2 -= learning_rate * delta_h1 * inputs[1]
    w3 -= learning_rate * delta_h2 * inputs[0]
    w4 -= learning_rate * delta_h2 * inputs[1]

    b1 -= learning_rate * (delta_h1 + delta_h2)

    # 每1000次迭代输出一次总误差
    if epoch % 1000 == 0:
        E_o1 = 0.5 * (expected_output[0] - a_o1) ** 2
        E_o2 = 0.5 * (expected_output[1] - a_o2) ** 2
        E_total = E_o1 + E_o2
        print(f'迭代 {epoch}, 总误差: {E_total}')

# 训练后的最终输出
z_h1, z_h2, a_h1, a_h2, z_o1, z_o2, a_o1, a_o2 = forward_propagation(inputs, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2)
print(f'训练后的最终输出: {[a_o1, a_o2]}')
print(f'最终总误差: {E_total}')
