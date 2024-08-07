import torch
# torch.rand 生成的是[0,1]均匀分布的随机数，而torch.randn生成的是标准正态分布的随机数
x = torch.randn((4, 4), requires_grad=True)
y = 2*x
z = y.sum()

print(z.requires_grad)  # True

y.backward()  # 计算z(标量)关于其所有需要梯度的输入的梯度,z关于x的梯度就是每个元素为2的张量
print(x.grad)