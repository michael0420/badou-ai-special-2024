import torch


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))

    # 定义全连接
    def forward(self, x):
        # mm-矩阵乘法  wx
        x = x.mm(self.weight)
        if self.bias:
            # wx + b
            x = x + self.bias.expand_as(x)
        return x

