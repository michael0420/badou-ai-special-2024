import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Model:
    # 定义网络结构 损失函数 优化器
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.cost(cost)
        self.optimizer = self.optimizer(optimist)

    def cost(self, cost):
        # 支持交叉熵和均方差 前者更多用于概率 后者更多用于数值
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    def optimizer(self, optimist, **rests):
        # 支持sgd adam rmsp做优化器
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            # 在每轮训练开始前 都把损失值置为0
            running_loss = 0.0
            # 读取数据并把梯度置零
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                # 前向传播 计算损失 并反向传播进行优化
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # 累加100次的损失值 并在一百次后打印平均值
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
        # 梯度下降
        with torch.no_grad():
            for data in test_loader:
                # 读取测试集数据
                images, labels = data
                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # 计算预测准确率
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def mnist_load_data():
    # 转化为tensor并进行标准化
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])

    #加载训练和测试集
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
    return trainloader, testloader

class MnistNet(torch.nn.Module):
    # 这里面主要包含要训练的参数
    def __init__(self):
        # 继承自MinistNet类
        super(MnistNet, self).__init__()
        # 第一层输入为28*28 输出为512
        self.fc1 = torch.nn.Linear(28*28, 512)
        # 第二层输入为512 输出也为512
        self.fc2 = torch.nn.Linear(512, 512)
        # 第三层输入为512 输出为10 因为只有十个种类
        self.fc3 = torch.nn.Linear(512, 10)

    # 这里包含没有训练的层
    def forward(self, x):
        # 先把特征拉直成一维数组
        x = x.view(-1, 28*28)
        # 过两层relu 最后一层输出概率值 过softmax
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

if __name__ == '__main__':
    # 神经网络类名为MnistNet
    net = MnistNet()
    # 损失函数为交叉熵 优化器为rmsp
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    # 加载数据
    train_loader, test_loader = mnist_load_data()
    # 训练和评估
    model.train(train_loader)
    model.evaluate(test_loader)