import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transformsn


class Model:

    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    # 选择什么样激活函数
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),     # 交叉熵
            'ESM': nn.MSELoss()
        }
        return support_cost[cost]

    # 选择什么样的优化向
    def create_optimizer(self, optimist, **reste):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, ** reste),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, ** reste),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, ** reste)
        }
        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()

                # forward + backward + optimize
                # 正向 + 反向 + 优化向
                outputs = self.net(inputs)
                loss = self.cost(outputs,labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 100 == 0:
                    # print(i,  len(train_loader))
                    print('epoch %d, %.2f%%, loss: %.3f' % (epoch+1, (i+1)*1.0/len(train_loader)*100, running_loss/100))
                    running_loss = 0.0
        print('训练完成')

    def evaluate(self, test_loader):
        print('测试')
        correct = 0
        total = 0
        # 在with下面的函数变量都不需要进行求导
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('%d%%' % (100 * correct / total))


def mnist_load_data():
    transform = transformsn.Compose(
        [transformsn.ToTensor(),
         transformsn.Normalize([0, ], [1, ])])

    trainset = torchvision.datasets.MNIST(root='../../../../data/mnist_data', train=True,
                                          download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='../../../../data/mnist_data', train=False,
                                          download=False, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                              shuffle=True, num_workers=2)
    return trainloader, testloader


class MnistNet(torch.nn.Module):
    """ 定义网络模型  """
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        # 类型reshapegongn
        x = x.view(-1, 28*28)
        # 激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)