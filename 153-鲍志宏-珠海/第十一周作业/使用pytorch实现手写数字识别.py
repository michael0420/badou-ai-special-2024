import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# 定义用于MNIST手写数字识别的神经网络
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        # 定义三层全连接神经网络
        self.fc1 = nn.Linear(28*28, 512)  # 输入层：28*28像素，输出层：512个神经元
        self.fc2 = nn.Linear(512, 512)    # 隐藏层：512个神经元，输出层：512个神经元
        self.fc3 = nn.Linear(512, 10)     # 隐藏层：512个神经元，输出层：10个神经元（对应0-9的手写数字）

    def forward(self, x):
        x = x.view(-1, 28*28)             # 将输入数据展平为一维
        x = F.relu(self.fc1(x))           # 应用ReLU激活函数
        x = F.relu(self.fc2(x))           # 应用ReLU激活函数
        x = F.log_softmax(self.fc3(x), dim=1)  # 应用Log Softmax激活函数
        return x

# 定义一个通用的Model类，包含初始化、训练和评估方法
class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)

    def create_cost(self, cost):
        # 定义支持的损失函数
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        # 定义支持的优化器
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        # 训练模型
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()    # 清零梯度

                outputs = self.net(inputs)    # 前向传播
                loss = self.cost(outputs, labels)  # 计算损失
                loss.backward()               # 反向传播
                self.optimizer.step()         # 更新参数

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[第%d轮, %.2f%%] 损失值: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print('训练结束')

    def evaluate(self, test_loader):
        # 评估模型
        print('评估中 ...')
        correct = 0
        total = 0
        with torch.no_grad():  # 测试时不需要梯度
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('测试图像准确性: %d %%' % (100 * correct / total))

# 加载MNIST数据集并进行预处理
def mnist_load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),               # 转换为Tensor
        transforms.Normalize((0.5,), (0.5,)) # 归一化
    ])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=True, num_workers=2)
    return trainloader, testloader

if __name__ == '__main__':
    net = MnistNet()                            # 实例化模型
    model = Model(net, 'CROSS_ENTROPY', 'RMSP') # 定义Model实例，使用交叉熵损失和RMSprop优化器
    train_loader, test_loader = mnist_load_data() # 加载训练和测试数据
    model.train(train_loader)                    # 训练模型
    model.evaluate(test_loader)                  # 评估模型
