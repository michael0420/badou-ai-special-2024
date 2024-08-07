import torch

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = x.mm(self.weight)
        if self.bias:
            x = x + self.bias.expand_as(x)
        return x

if __name__ == '__main__':
    # train for mnist
    net = Linear(3,2)
    x = net.forward
    print('11',x)
