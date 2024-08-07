import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def minist_load_data():  # 首先写读取数据集函数
    """
    transforms.Compose将多个转换步骤组合在一起。
    transforms.ToTensor()：将PIL图像或NumPy ndarray转换为PyTorch张量（Tensor），并自动将像素值从[0, 255]缩放到[0.0, 1.0]。
    transforms.Normalize的参数主要包括两个：mean和std，都是一个序列，包含每个通道要减去均值&除以的标准差
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0, ], [1, ])])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    # 封装训练集，设置批大小为32，启用打乱数据(每次训练迭代时都会打乱训练数据的顺序，防作弊)，并使用2个工作进程来加速数据加载。
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader


class MnistNet(torch.nn.Module):  # 定义网络结构
    def __init__(self):
        super().__init__()  # python3里super无需传入参数
        self.fc1 = nn.Linear(28*28, 512)  # 定义全连接层
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        """
        x（通常是一个四维张量，形状为 [batch_size, channels, height, width]
        但在这里因为处理的是灰度图像，所以channels为 1
        全连接层期望的输入是一维或二维张量。
        如果处理彩色图像，并且每张图像有 3 个通道（红、绿、蓝），需要将图像展平为 [batch_size, 3*28*28]。
        """
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # x[batch_size, hidden_nodes]=[32, 512]
        x = F.softmax(self.fc3(x), dim=1)  # 设置dim=1，确保softmax在正确维度上计算
        return x


class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        if cost not in support_cost:
            raise ValueError(f"Unsupported cost:{cost}. Supported costs are:{list(support_cost.keys())}")
        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        """
        1.**:关键字参数解包可以动态地指定多个参数，而不需要在函数定义中显式地列出它们这非常有用，因为不同的优化器可能有不同的参数
        例如Adam优化器支持betas和weight_decay等参数，而 SGD 优化器可能只支持momentum和weight_decay
        2.self.net.parameters()是torch.nn.Module类的一个内置方法,作用是返回一个包含模型（即 self.net）中所有可训练参数的迭代器。
        这些参数是神经网络在训练过程中需要被优化的对象，通常包括权重（weights）和偏置（biases）。
        在 PyTorch 中，每个 nn.Parameter 对象都自动注册到其所属的 nn.Module中，因此当你调用 self.net.parameters()时，它会遍历整个模型并收集所有这样的参数。
        这些参数随后可以被传递给优化器（如 optim.SGD, optim.Adam, optim.RMSprop等），以便在训练过程中进行更新。
        """
        support_optimist = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
            }
        if optimist not in support_optimist:
            raise ValueError(f"Unsupported optimizer:{optimist}. Supported optimizers are:{list(support_optimist.keys())}")
        return support_optimist[optimist]

    def train(self, train_loader, epochs=3):
        for epoch in range(epochs):
            running_loss = 0.0  # 初始化，用于累加当前epoch的损失，以计算平均损失
            # enumerate 函数用于同时获取数据批次的索引（i）和数据本身（data）。
            for i, data in enumerate(train_loader, 0):  # 0(默认值)是一个可选的start参数，它指定了枚举的起始索引值.
                inputs, labels = data  # train_loader迭代产生的每个data元素实际上是一个元组（tuple），这个元组包含了当前批次（batch）的输入数据和对应的标签。
                self.optimizer.zero_grad()  # 清除（即归零）之前所有参数的梯度,以便只包含当前批次数据的梯度信息

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()  # 计算损失关于模型参数的梯度
                self.optimizer.step()  # 根据计算得到的梯度更新模型参数

                running_loss += loss.item()  # 为了节省性能loss返回的是一个标量，需要用.item()转换为Python的数值类型才可参与计算
                if i % 100 == 0:  # 每100个batch
                    # 打印当前epoch，当前已经处理的数据占整个训练集的百分比以及每100个批次的平均损失
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch+1, (i+1)*1./len(train_loader)*100, running_loss/100))
                    running_loss = 0.0
        print('Finished training')

    def evaluate(self, test_loader):
        print('Evaluating')
        correct = 0  # 用于记录预测正确的样本数
        total = 0  # 用于记录总样本数
        with torch.no_grad():  # 评估不需要计算梯度，用上下文管理器禁用梯度可以减少内存消耗并加速计算
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)  # 1 表示在第二个维度（通常是类别维度）上应用 argmax
                total += labels.size(0)  # 用来获取当前批次中图像的数量
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d%%' % (correct/total*100))


if __name__ == '__main__':
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = minist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)


































