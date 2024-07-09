import torch
import torch.nn as nn  # 封装好的类，继承自module
import torch.optim as optim
import torch.nn.functional as F
import torchvision  # 视觉化
import torchvision.transforms as transforms  # 数据预处理工具


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
        }  # 字典
        return support_cost[cost]  # 查字典

    def create_optimizer(self, optimist, **rests):  # **rest:输入的其他参数项
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):  # enumerate(iterable,start=0) 输出结果是i,data 从0开始，如果是1，序号从1开始
                inputs, labels = data

                self.optimizer.zero_grad()  # 清除优化器关于所有参数x的累计梯度，一般在loss.backward前使用

                # forward -> backward -> optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()  # 将损失向输入侧进行反向传播，计算所有变量的梯度值
                self.optimizer.step()  # 优化，加学习率

                # 展示计算完成的过程
                running_loss += loss.item()  # item:输出具体元素值
                if i % 100 == 0:  # 每100个，打印一次信息
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, 100. * (i + 1) / len(train_loader), running_loss / 100))
                    running_loss = 0.0  # 每100次计算一次平均损失，然后重置为0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating...')
        correct = 0
        total = 0
        with torch.no_grad():  # 测试，评估，不需要梯度计算
            for data in test_loader:  # 不需要i了，就不用enumerate了
                images, labels = data  # 图的数据，标签

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)  # argmax：指定维度的最大值（列的维度）
                total += labels.size(0)  # 返回当前批次的样本数。每处理一个批次，total就增加该批次的样本数。(32每批)

                correct += (predicted == labels).sum().item()  # 若正确加1，不正确加0

        print('Accuracy of the network : %d %%' % (100. * correct / total))


def mnist_load_data():
    transform = transforms.Compose(  # 结合括号中方法
        [transforms.ToTensor(),  # 将图片转换成tensor类型   从whc 转换到cwh
         # transforms.Normalize([0,], [1,])
         transforms.Normalize(0, 1)])  # (channel_mean,channel_std)  均值0，方差1
    '''
    torchvision.datasets.MNIST()
        root: 根，,NIST所在的文件夹
        train： 若为true 从training.pt创建数据集，否则从test.pt创建
        download: true 从网络下载数据
        transform(): 接受PIL图像并返回已转换版本的函数

    torch.utils.data.DataLoader(）
       dataset (Dataset) – 要从中加载数据的数据集。
       batch_size (int, optional) – 每批次要装载多少样品 
       shuffle (bool, optional) – 设置为True以使数据在每个时期都重新洗牌 
       sampler (Sampler or Iterable, optional) – 定义从数据集中抽取样本的策略
       batch_sampler (Sampler or Iterable, optional) – 类似于采样器，但一次返回一批索引。 与batch_size，shuffle，sampler和drop_last互斥。 
       num_workers (int, optional) – 多少个子流程用于数据加载（linux）。 0表示将在主进程中加载数据。 （默认值：0  根据电脑性能设置。）      
    '''

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=0)

    return trainloader, testloader


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()  # super init 调用父类（torch.nn.Model）
        self.fc1 = nn.Linear(28 * 28, 512)  # 具备学习参数，用nn.model方便，不用再自行配置w，b
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # view:修改shape
        x = F.relu(self.fc1(x))  # 不具备学习参数，用function
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)  # dim=1 行不变，列之间比较
        return x


if __name__ == '__main__':
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)