# 导入 torch包
import torch

# 创建tensor x , requires_grad 自动求带，默认值false
x = torch.randn((4, 4), requires_grad=True)
y = 2 * x
z = y.sum()

# 这里只有一个变量x， 而且后面进行的常量操作， 所以requires_grad为true
print(z.requires_grad)  # True

# 反向传播(梯度求导，权值更新)
z.backward()

print(x.grad)
'''
tensor([[ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.]])
'''
