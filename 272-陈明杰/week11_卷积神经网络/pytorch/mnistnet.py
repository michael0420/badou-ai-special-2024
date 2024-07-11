import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# # def mnist_load_data():
# #     # 定义一个名为mnist_load_data的函数，这个函数没有输入参数，并且会返回训练集和测试集的数据加载器。
# #
# #     transform = transforms.Compose(
# #         # 定义一个数据预处理流程，该流程是由多个转换步骤组成的。
# #         [transforms.ToTensor(),  # 第一个转换步骤是将图像转换为PyTorch张量，并且像素值从[0, 255]缩放到[0.0, 1.0]。
# #          transforms.Normalize([0, ], [1, ])])  # 第二个转换步骤是标准化，但这里的参数设置似乎不正确，因为通常标准化需要均值和标准差两个参数列表。
# #
# #     # 加载MNIST训练集
# #     trainset = torchvision.datasets.MNIST(root='./data', train=True,  # 指定数据集存储在'./data'目录下，并且加载的是训练集。
# #                                           download=True,  # 如果数据集不存在，则下载它。
# #                                           transform=transform)  # 应用前面定义的数据预处理流程。
# #
# #     # 为训练集定义一个数据加载器
# #     trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,  # 从训练集中加载数据，每个批次包含32个样本。
# #                                               shuffle=True,  # 在每个训练周期开始时，打乱训练集中的样本顺序。
# #                                               num_workers=2)  # 使用两个子进程来并行加载数据。
# #
# #     # 加载MNIST测试集
# #     testset = torchvision.datasets.MNIST(root='./data', train=False,  # 指定数据集存储在'./data'目录下，并且加载的是测试集。
# #                                          download=True,  # 如果测试集不存在，则下载它（但实际上，如果训练集已下载，测试集通常已包含在内）。
# #                                          transform=transform)  # 同样应用前面定义的数据预处理流程。
# #
# #     # 为测试集定义一个数据加载器
# #     testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True,
# #                                              num_workers=2)  # 加载测试集数据，但通常这里shuffle应该为False，因为我们不需要在测试时打乱数据顺序。
# #
# #     # 返回训练集和测试集的数据加载器
# #     return trainloader, testloader
#
# class Model:
#     def __init__(self, net, cost, optimist):
#         # 神经网络
#         self.net = net
#         # 损失函数
#         self.cost = self.create_cost(cost)
#         # 优化算法
#         self.optimist = self.create_optimist(optimist)
#
#     # 创建损失函数
#     def create_cost(self, cost):
#         if cost == 'CROSS_ENTROPY':
#             return nn.CrossEntropyLoss()
#         if cost == 'MSE':
#             return nn.MSELoss()
#         pass
#
#     # **rests参数相当于可变参数列表，可以不传或者传多个参数
#     # 在PyTorch中，optim.SGD 是用于实现随机梯度下降（Stochastic Gradient Descent, SGD）的优化器。
#     # 当你使用 optim.SGD(self.net.parameters(), lr=0.1, **rests) 这样的代码时，你正在配置SGD优
#     # 化器来更新模型 self.net 的参数。这里的参数有其特定的含义：
#     #
#     # self.net.parameters()：这是你要优化的模型参数。self.net 是一个 torch.nn.Module 的实例（例
#     # 如你的 MnistNet 类的一个实例），它包含了模型的权重和偏置。parameters() 方法返回一个生成器，该
#     # 生成器产生模型中的所有参数（权重和偏置）。
#     # lr=0.1：这是学习率（learning rate），它决定了在每次更新时参数移动的步长。较高的学习率通常意味
#     # 着参数更新的步长较大，可能导致模型训练更快，但也可能导致模型在最优解附近震荡而无法收敛。相反，较
#     # 低的学习率可能导致模型训练较慢，但可能更稳定。
#     # **rests：这是一个Python字典解包操作。假设你有一个字典 rests，它包含了除学习率之外的其他SGD优
#     # 化器的配置参数，**rests 会将这些参数传递给 optim.SGD。这允许你以灵活的方式配置优化器，而无需
#     # 在函数调用中显式指定每个参数。
#
#     # 一些常见的SGD优化器参数（除了学习率）包括：
#     # momentum：动量（Momentum）是SGD的一个变种，它模拟了物理世界中的动量概念，帮助加速SGD在相关方
#     # 向上的搜索并抑制震荡。默认值为0，表示不使用动量。
#     # dampening：动量的阻尼因子，用于控制动量的衰减。默认值为0。
#     # weight_decay：L2正则化项的系数（也称为权重衰减）。这有助于防止模型过拟合。默认值为0，表示不使
#     # 用L2正则化。
#     # nesterov：是否使用Nesterov动量。如果设置为True，则使用Nesterov动量。默认值为False。
#
#     # 这是一个优化器
#     def create_optimist(self, optimist, **rests):
#         support_optim = {
#             'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
#             'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
#             'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
#         }
#         return support_optim[optimist]
#         pass
#
#     # 训练
#     def train(self, train_loader, epoches=3):
#         for epoch in range(epoches):
#             running_loss = 0.0
#             # 遍历train_loader对象，i从0开始，因为后面设置了0
#             for i, data in enumerate(train_loader, 0):
#                 input, label = data
#                 # 前向传播
#                 output = self.net(input)
#                 # 计算损失值
#                 loss = self.cost(output, label)
#                 # 计算反向传播的权重值
#                 loss.backward()
#                 # 更新反向传播的权重
#                 self.optimist.step()
#                 # 清空梯度信息，避免影响下一次
#                 self.optimist.zero_grad()
#
#                 running_loss += loss.item()
#                 if i % 100 == 0:
#                     print('[epoch %d, %.2f%%] loss: %.3f' %
#                           (epoch + 1, (i + 1) * 1. / len(train_loader)*100, running_loss / 100))
#                     running_loss = 0.0
#
#         print('Finished Training')
#         pass
#
#     # 推理
#     def evaluate(self, test_loader):
#         total = 0
#         correct = 0
#         # 声明不需要自动微分
#         with torch.no_grad():
#             for data in test_loader:
#                 input, label = data
#                 output = self.net(input)
#                 # 过softmax函数转化为概率
#                 output = F.softmax(output, dim=1)
#                 # 返回最大概率的index，res是一个数组，因为每次有batch_size=32，所以
#                 # res有32个元素，每个元素是[0,9]的下标
#                 res = torch.argmax(output, 1)
#                 # label有多少行，total就加多少
#                 total += label.size(0)
#                 # 对比res和label的对应位置，如果相等就是true，不等就是false，
#                 # 所以res == label返回一个bool数组，.sum()求和说明这一个batch_size
#                 # 元素中有sum个预测正确，那么.item()是转化为整数，+=到correct中，用于
#                 # 求出正确率
#                 correct += (res == label).sum().item()
#         print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
#
#
# def mnist_load_data():
#     # 数据的转换方式
#     transform = transforms.Compose([transforms.ToTensor()
#                                        , transforms.Normalize((0.1307,), (0.3081,))])
#     # transform = transforms.Compose([transforms.ToTensor()
#     #                                    , transforms.Normalize([0, ], [1, ])])
#
#     # 读取训练数据集，根据数据的转化方式，把读取到的数据转化为张量，并且归一化到[0,1]区间
#     train_set = torchvision.datasets.MNIST(root='./data', train=True,
#                                            download=True, transform=transform)
#     # shuffle表示是否打乱顺序，num_workers表示线程数
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=32
#                                                , shuffle=True, num_workers=2)
#     test_set = torchvision.datasets.MNIST(root='./data', train=False,
#                                           download=True, transform=transform)
#     test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,
#                                               shuffle=False, num_workers=2)
#     return train_loader, test_loader
#
#
# class MnistNet(torch.nn.Module):
#     def __init__(self):
#         # 初始化父类
#         super(MnistNet, self).__init__()
#         # 输入层到隐藏层1
#         self.fc1 = torch.nn.Linear(28 * 28, 512)
#         # 隐藏层1到隐藏层2
#         self.fc2 = torch.nn.Linear(512, 512)
#         # 隐藏层2到输出层
#         self.fc3 = torch.nn.Linear(512, 10)
#
#     def forward(self, x):
#         # x = x.reshape(-1, 28 * 28)
#         # print(x.shape)
#         # 转化成训练所需要的正确的格式
#         x = x.view(-1, 28 * 28)
#         # print(x.shape)
#         # 输入层到隐藏层1，过激活函数relu
#         x = F.relu(self.fc1(x))
#         # 隐藏层1到隐藏层2，过激活函数relu
#         x = F.relu(self.fc2(x))
#         # 因为是多分类问题，所以输出层应该过softmax(这是特殊的激活函数，用于多分类问题)函数，
#         # 无需过relu激活函数
#         # x = F.softmax(self.fc3(x), dim=1)
#         # 当我们使用的损失函数是交叉熵函数的时候，forward函数返回的
#         # 是函数的原始输出值，因为在后面计算损失值的交叉熵函数中本身已经
#         # 包含了softmax函数的实现，所以这里无需调用softmax函数，
#         # 如果损失函数不是交叉熵，那么根据具体情况决定是否需要调用softmax函数
#
#         # 隐藏层2到输出层，这里可以直接返回输出层的原始结果，因为后面的交叉熵函数中
#         # 就已经包含了softmax函数
#         x = self.fc3(x)
#         return x
#
#
# if __name__ == '__main__':
#     # 构建一个全连接层的网络
#     net = MnistNet()
#     # 添加一些优化参数，构建模型
#     # 损失函数为交叉熵函数，优化项为RMSP
#
#     # RMSP代表RMSprop优化算法。RMSprop是一种用于梯度下降（包括mini-batch梯度下降）的
#     # 算法，通过调节每个参数的学习率来消除摆动幅度大的方向，并加速在摆动幅度小的方向的移
#     # 动，从而使训练过程更快地进行。RMSprop算法在AdaGrad算法的基础上进行了改进，通过引
#     # 入一个衰减率来控制历史梯度的影响，使得算法在更新参数时更加注重于近期的梯度，从而避
#     # 免了AdaGrad算法中学习率过早减小的问题。
#     # RMSprop的优化公式可以分为两种，一种是不加Momentum的RMSprop更新梯度公式，另一种是
#     # 加了Momentum的RMSprop更新梯度公式。两者可以同时应用，其中Momentum的惯性公式主要
#     # 针对更新的梯度参数，而RMSprop的惯性公式主要针对二阶动量。
#     model = Model(net, 'CROSS_ENTROPY', 'RMSP')
#     # 读取数据
#     train_loader, test_loader = mnist_load_data()
#     # 训练
#     model.train(train_loader)
#     # 推理
#     model.evaluate(test_loader)
#


# class Model:
#     def __init__(self, net, cost, optimist):
#         # 神经网络部分
#         self.net = net
#         # 损失函数
#         self.cost = self.create_cost(cost)
#         # 优化器
#         self.optim = self.create_optim(optimist)
#
#     def create_cost(self, cost):
#         cost_hash = {
#             'CROSS_ENTROPY': nn.CrossEntropyLoss(),
#             'MSE': nn.MSELoss()
#         }
#         return cost_hash[cost]
#         pass
#
#     def create_optim(self, optimist, **rests):
#         optim_hash = {
#             "SGD": optim.SGD(self.net.parameters(), lr=0.1, **rests),
#             'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
#             'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
#         }
#         return optim_hash[optimist]
#         pass
#
#     def train(self, train_loader, epoches=3):
#         for epoch in range(epoches):
#             running_loss = 0.0
#             for i, data in enumerate(train_loader, 0):
#                 input, label = data
#                 # 前向传播
#                 x = self.net(input)
#                 # 求出损失值（误差）
#                 loss = self.cost(x, label)
#                 # 误差反向传播
#                 loss.backward()
#                 # 更新权重
#                 self.optim.step()
#                 # 清空梯度信息，避免影响下一次
#                 self.optim.zero_grad()
#
#                 running_loss += loss.item()
#                 if i % 100 == 0:
#                     print('[epoch %d, %.2f%%] loss: %.3f' %
#                           (epoch + 1, (i + 1) * 1. / len(train_loader) * 100, running_loss / 100))
#                     running_loss = 0.0
#
#         print('Finished Training')
#         pass
#
#     def evaluate(self, test_loader):
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for data in test_loader:
#                 input, label = data
#                 # 前向传播
#                 x = self.net(input)
#                 # 过softmax函数得到概率分布的数组
#                 x = F.softmax(x, dim=1)
#                 # 求出最大概率的下标
#                 index = torch.argmax(x, 1)
#                 total += label.size(0)
#                 correct += (index == label).sum().item()
#         print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
#         pass

class Model(nn.Module):
    def __init__(self, net, cost, optimist):
        super(Model, self).__init__()
        self.net = net
        self.cost = nn.CrossEntropyLoss()
        self.optim = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

    def train(self, data_loader, epochs=3):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, tmp in enumerate(data_loader, 0):
                data, label = tmp
                # 正向推导
                output = self.net(data)
                # 计算损失
                loss = self.cost(output, label)
                # 误差反向传播
                loss.backward()
                # 更新权重
                self.optim.step()
                # 清楚梯度，避免影响下一次
                self.optim.zero_grad()

                running_loss += loss.item()

                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader) * 100, running_loss / 100))
                    running_loss = 0.0

    def evaluate(self, data_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for i,tmp in enumerate(data_loader,0):
                data, label = tmp
                output = self.net(data)
                output = F.softmax(output,dim=1)
                index = torch.argmax(output,dim=1)
                total += label.size(0)
                correct += (index == label).sum().item()
            print('Accuracy of the network on the test '
                  'images: %d %%' % (100 * correct / total))
        pass


# class MnistNet(torch.nn.Module):
#     def __init__(self):
#         # 初始化父类
#         super(MnistNet, self).__init__()
#         # 输入层到隐藏层1
#         self.fc1 = torch.nn.Linear(28 * 28, 512)
#         # 隐藏层1到隐藏层2
#         self.fc2 = torch.nn.Linear(512, 512)
#         # 隐藏层2到输出层
#         self.fc3 = torch.nn.Linear(512, 10)
#
#     def forward(self, input):
#         # 这里一定要先转换格式，否则会报错
#         input = input.view(-1, 28 * 28)
#         # 输入层到隐藏层1，过激活函数
#         x = F.relu(self.fc1(input))
#         # 隐藏层1到隐藏层2，过激活函数
#         x = F.relu(self.fc2(x))
#         # 隐藏层2到输出层
#         x = self.fc3(x)
#         return x

class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        # 输入层到隐藏层1
        self.hidden1 = nn.Linear(in_features=28 * 28, out_features=512)
        # 隐藏层1到隐藏层2
        self.hidden2 = nn.Linear(in_features=512, out_features=512)
        # 隐藏层2到输出层
        self.output = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        hidden1 = F.relu(self.hidden1(x))
        hidden2 = F.relu(self.hidden2(hidden1))
        output = self.output(hidden2)
        return output


# def mnist_load_data():
#     # 定义转化方式
#     transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0, ], [1, ])])
#     # 加载train_set
#     train_set = torchvision.datasets.MNIST(root='./data', train=True,
#                                            download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=32
#                                                , shuffle=True, num_workers=2)
#     # 加载test_set
#     test_set = torchvision.datasets.MNIST(root='./data', train=True,
#                                           download=True, transform=transform)
#     test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,
#                                               shuffle=False, num_workers=2)
#     return train_loader, test_loader
#     pass
def mnist_load_data():
    tranform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0, ], [1, ])])
    train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                           download=True, transform=tranform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                               shuffle=True, num_workers=2)
    test_set = torchvision.datasets.MNIST(root='./data', train=True,
                                          transform=tranform, download=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,
                                              shuffle=False, num_workers=2)
    return train_loader, test_loader


if __name__ == '__main__':
    net = MnistNet()
    train_loader, test_loader = mnist_load_data()
    model = Model(net, 'CROSS_ENTROPY', 'SGD')
    model.train(train_loader)
    model.evaluate(test_loader)

# import numpy as np
#
# x=np.array([[1,2,3],
#             [4,5,6],
#             [7,8,9]])
# print(x.mean(axis=1))
