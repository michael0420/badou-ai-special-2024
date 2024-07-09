import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Model:
    # net-网络模型  cost-损失函数  optimist-优化项
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),  # 交叉熵
            'MSE': nn.MSELoss()  # MSE
        }

        return support_cost[cost]

    # 优化项本身可选
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),  # 随机梯度下降 精度要求不高
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),  # 先做momentum再做RMSP 收敛速度慢
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
            # momentum解决SGD优化算法摆动幅度大的问题  RMSP进一步解决百度幅度大的问题  收敛速度快
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)  # 正向过程:  输入  经过正向过程  输出 （softmax结果）
                loss = self.cost(outputs, labels)  # 计算损失函数
                loss.backward()  # 反向传播
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict  推理时不需要计算导数,所有变量、函数都不需要求导
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)  # 正向过程:  输入  经过正向过程  输出 （softmax结果）
                predicted = torch.argmax(outputs, 1)  # argmax 排序 取概率最高的
                total += labels.size(0)  # 统计推理了多少张图
                correct += (predicted == labels).sum().item()  # 统计正确率

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0, ], [1, ])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader


# 网络模型的构建
class MnistNet(torch.nn.Module):
    # 定义需要训练的层
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    # 定义参数不需要训练的层
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # reshape
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

    # backward 反向自动求导


if __name__ == '__main__':
    # train for mnist
    # [1] 写好网络
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    # [2] 编写好数据的标签和路径索引
    train_loader, test_loader = mnist_load_data()
    # [3] 把数据送到网络
    model.train(train_loader)
    model.evaluate(test_loader)
