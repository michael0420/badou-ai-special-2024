import torch

# 创建一个4*4的元组，并初始化为随机数，并把自动微分功能打开
x = torch.randn((4, 4), requires_grad=True)
print(x)
y = 2 * x
z = y.sum()

z.backward()
print(z)
print(x.grad)
