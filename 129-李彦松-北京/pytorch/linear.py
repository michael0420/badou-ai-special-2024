import torch

# 定义一个线性层类，继承自torch.nn.Module
class Linear(torch.nn.Module):
# 初始化函数，接收输入特征数、输出特征数和偏置项是否存在的参数
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()  # 调用父类的初始化函数
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))  # 初始化权重参数
        if bias:  # 如果存在偏置项
            self.bias = torch.nn.Parameter(torch.randn(out_features))  # 初始化偏置参数

    # 前向传播函数，接收输入x
    def forward(self, x):
        x = x.mm(self.weight)  # 输入x与权重进行矩阵乘法
        if self.bias:  # 如果存在偏置项
            x = x + self.bias.expand_as(x)  # 将偏置项添加到结果上
        return x  # 返回结果

if __name__ == '__main__':
    # 创建一个线性层实例，输入特征数为3，输出特征数为2
    net = Linear(3,2)
    x = net.forward  # 获取前向传播函数
    print('11',x)  # 打印前向传播函数


# class Linear(torch.nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super(Linear, self).__init__()
#         self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
#         if bias:
#             self.bias = torch.nn.Parameter(torch.randn(out_features))
#
#     def forward(self, x):
#         x = self.weight.mv(x)  # 修改这里
#         if self.bias is not None:
#             x = x + self.bias.expand_as(x)
#         return x
#
# if __name__ == '__main__':
#     net = Linear(3,2)
#     x_input = torch.randn(3)  # 创建一个随机的3维向量作为输入
#     x_output = net(x_input)  # 调用forward方法并获取返回值
#     print('11', x_output)  # 打印返回值