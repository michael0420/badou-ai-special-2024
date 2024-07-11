import torch

# 自定义神经网络
class Net(torch.nn.Module):
    inNodes = None
    def __init__(self, inNodes, hdNodes, outNodes):
        super(Net, self).__init__()
        self.inNodes = inNodes
        self.inputLevel = torch.nn.Linear(inNodes, hdNodes)
        self.hidenLevel = torch.nn.Linear(hdNodes, outNodes)

    def forward(self, inM):
        inM = inM.view(-1, self.inNodes)
        inM = torch.nn.functional.relu(self.inputLevel(inM))
        inM = torch.nn.functional.relu(self.hidenLevel(inM))
        inM = torch.nn.functional.softmax(inM, dim=1)
        return inM

if __name__ == "__main__":
    myNet = Net(28 * 28, 512, 10)
    import numpy as np
    inM = torch.tensor(np.zeros([28 * 28, 512]))
    myNet.forward(inM)