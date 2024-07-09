import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Model:
    def __init__(self, net, cost, optimist):#net:神经网络，cost:损失函数，optimist:优化器
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(), # 交叉熵
            'MSE': nn.MSELoss() # 均方误差
        }

        return support_cost[cost]

#parameters()返回一个生成器，该生成器产生模型的所有参数（权重和偏置）。这些参数是模型训练过程中需要优化的值。
    def create_optimizer(self, optimist, **rests): # **rests:可变参数，lr:学习率
        support_optim = { # **rests 是一个可变参数，它可以接收任意数量的关键字参数。在这个方法中，**rests 用于接收优化器的额外参数。
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=3): # epoches:训练次数,train_loader:训练数据
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def mnist_load_data(): # 加载mnist数据集
    transform = transforms.Compose( # transforms.Compose()将多个transform组合起来使用
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])#transforms.ToTensor() 是一个变换，它将 PIL Image 或者 numpy.ndarray 转换为 torch.Tensor，并且缩放图像的像素强度值到 [0., 1.]。  transforms.Normalize([0,], [1,]) 是一个变换，它对图像进行标准化。这个变换需要两个参数：mean 和 std，分别代表各个通道的均值和标准差。在这个例子中，均值和标准差都是 [0,] 和 [1,]，这意味着这个变换实际上并没有改变图像的像素值。

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2,drop_last=True)
    return trainloader, testloader


class MnistNet(torch.nn.Module):
    def __init__(self): # 定义网络结构,输入层为28*28，输出层为10,隐藏层为512
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28) # view()函数作用是将一个多行的Tensor拼接成一行
        x = F.relu(self.fc1(x)) # F.relu()是一个激活函数，它的作用是将输入的值限制在 0 和正无穷之间。这个函数的公式是 relu(x) = max(0, x)。
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1) # F.softmax()是一个激活函数，它的作用是将输入的值转换为概率。这个函数的公式是 softmax(x) = exp(x) / exp(x).sum()。
        return x

if __name__ == '__main__':
    # train for mnist
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
