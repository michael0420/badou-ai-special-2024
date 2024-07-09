# 导入PyTorch相关库
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Model:  # 定义模型类，用于封装网络模型、损失函数和优化器
    def __init__(self, net, cost, optimist):  # 初始化模型，包括网络、损失函数和优化器

        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)

    def create_cost(self, cost):  # 根据给定的损失函数类型创建损失函数实例
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):  # 根据给定的优化器类型创建优化器实例
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]

    def train(self, data_loader, epoches=2):
        """
        训练模型
        :param data_loader: 训练数据加载器
        :param epoches: 训练轮数
        """
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(
                    data_loader):  # data_loader 是一个 DataLoader 对象，它负责从训练数据集中批量取出数据。在每次迭代中，data 就是这批数据，通常是一个包含数据和标签的元组。
                inputs, labels = data

                self.optimizer.zero_grad()  # 将模型中所有可学习参数（如权重和偏置）的梯度清零。

                # 前向传播、计算损失、反向传播、优化
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)  # 计算预测输出与真实标签之间的损失（loss）。
                loss.backward()  # 反向传播
                self.optimizer.step()  # 基于当前的梯度信息以及所选择的优化算法（如SGD、Adam等）来更新网络中的所有可学习参数

                running_loss += loss.item()  # 累加当前批次的损失值到running_loss变量中，以便跟踪整个训练过程中的平均损失或累计损失。
                if i % 100 == 0:  # 每100条数据打印一次信息
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(data_loader), running_loss / 100))
                    running_loss = 0.0

            print('Finished Training Epoch %d' % (epoch + 1))

    def evaluate(self, test_data_loader):
        """
        评估模型在测试集上的性能
        :param test_data_loader: 测试数据加载器
        """
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # 表示测试的过程中不需要计算梯度
            for data in test_data_loader:
                images, labels = data
                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)  # 1表示取top1，即取到概率最高的索引

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

    def query(self, pic):  # 对单张图片进行预测
        image, label = pic
        result = self.net(image)
        predicted = torch.argmax(result, 1)
        print('[Query result is  %d, real number is  %d]' % (predicted, label))

        import matplotlib.pyplot as plt
        plt.imshow(image.squeeze(), cmap='gray')  # squeeze()用于去除单通道图像的unsqueeze维度，cmap='gray'是因为MNIST是灰度图
        plt.title(f"Label: {label}")
        plt.show()


class MnistNet(torch.nn.Module):  # 定义用于MNIST数据集的网络模型
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):  # 网络的正向传播过程

        x = x.view(-1, 28 * 28)  # x.view的作用是reshape，-1表示这个维度不变
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


# 加载MNIST数据集
def mnist_load_data():  # 加载并处理MNIST数据集

    # 将图像转换为Tensor并进行归一化
    transform = transforms.Compose([
        # 将图像数据转换为Tensor格式
        transforms.ToTensor(),
        # 对Tensor数据进行归一化，使像素值位于0到1之间
        transforms.Normalize([0, ], [1, ])
    ])

    # 加载MNIST训练集
    # root参数指定数据集下载后的保存路径；train参数设置为True表示加载训练集； download参数设置为True表示如果数据集不存在则自动下载； transform参数指定对数据进行的转换操作
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)

    # 使用DataLoader加载训练集，以便进行批量训练；  batch_size参数指定每个批次的大小；  shuffle参数设置为True表示训练时打乱数据顺序； num_workers参数指定用于数据加载的线程数
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    # 加载MNIST测试集；    train参数设置为False表示加载测试集
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    # 使用DataLoader加载测试集，用于模型的评估
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    return trainloader, testloader, trainset[random.randint(0, 100)]


if __name__ == '__main__':
    net = MnistNet()  # 创建模型结构
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')  # 创建模型实例
    data_loader, test_loader, random_pic = mnist_load_data()  # 加载数据
    model.train(data_loader)  # 训练模型
    model.evaluate(test_loader)  # 模型测试
    model.query(random_pic)  # 随机图片推理
