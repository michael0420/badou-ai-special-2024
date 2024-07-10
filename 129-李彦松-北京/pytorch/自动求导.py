import torch

x = torch.randn((4,4), requires_grad=True)
y = 2*x
z = y.sum()

print (z.requires_grad)  # True

z.backward()

print (x.grad)
'''
tensor([[ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.],
        [ 2.,  2.,  2.,  2.]])
'''
