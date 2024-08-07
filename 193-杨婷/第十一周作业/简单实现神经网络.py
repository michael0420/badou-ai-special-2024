import torch

'''
torch.nn.Module是所有神经网络模块的基类。
当你想要定义一个自己的神经网络层时，通常会通过继承torch.nn.Module类来实现
'''


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()  # 调用父类 torch.nn.Module 的构造函数,以确保Module的基础初始化逻辑被执行
        '''
        torch.nn.Parameter被赋值给一个nn.Module的属性时，它会自动被注册为一个模型参数。
        这意味着PyTorch的优化器（如torch.optim.SGD或torch.optim.Adam）能够追踪这个Parameter，并在训练过程中更新它的值。

        在定义自定义神经网络层或模块时，通常会使用torch.nn.Parameter来创建权重和偏置等可学习参数。
        这样做的好处是，PyTorch能够自动管理这些参数的梯度，并在调用.backward()时计算它们。
        '''
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        # 权重矩阵的每一行都对应于输出特征空间中的一个特征，而每一列都对应于输入特征空间中的一个特征
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))
        print(self.weight.shape)

a = Linear(3,2)

print(a)