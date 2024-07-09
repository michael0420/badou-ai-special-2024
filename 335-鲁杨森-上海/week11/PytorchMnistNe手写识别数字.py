'''
@author luyangsen
MnistNet Pytorch手写数字识别
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class MnistNet:
    def __init__(self, cost, opt):
        """
        定义模型结构,选择损失函数和优化器
        :param cost: 损失函数，仅支持CROSS_ENTROPY和MSE
        :param opt: 优化器，仅支持SGD、ADAM、RMSP
        """
        self.mnist_model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax()
        )
        support_cost = {'CROSS_ENTROPY': nn.CrossEntropyLoss(), 'MSE': nn.MSELoss()}
        support_optim = {'SGD': optim.SGD(self.mnist_model.parameters(), lr=0.1),
                         'ADAM': optim.Adam(self.mnist_model.parameters(), lr=0.01),
                         'RMSP': optim.RMSprop(self.mnist_model.parameters(), lr=0.001)}
        self.cost = support_cost[cost]
        self.optimizer = support_optim[opt]

    def train(self, train_loader, epochs=3):
        """
        模型训练
        :param train_loader: 包含训练数据和标签数据
        :param epochs: 代
        :return:
        """
        print('start training...')
        for e in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                train_data, labels = data
                self.optimizer.zero_grad()  # 清空梯度
                train_data = train_data.view(-1, 28 * 28)
                outputs = self.mnist_model(train_data)
                loss = self.cost(outputs, labels)
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数
                running_loss += loss.item()
                if i % 100 == 0:
                    print(f'epoch{e} {float(i + 1) / len(train_loader)} loss:{running_loss}')
                    running_loss = 0.0
        print('training fished...')

    def evaluate(self, test_loader):
        """
        使用测试数据集评估模型的泛化能力
        :param test_loader: 测试数据集
        :return:
        """
        print('start evaluate...')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                test_data, labels = data
                test_data = test_data.view(-1, 28 * 28)
                outputs = self.mnist_model(test_data)
                predictions = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        print(f'Accuracy on test data is: {correct / total}')
        print('evaluate finished...')


def data_loader():
    """
    加载mnist数据集，并进行乱序和封装成批次处理
    :return: 返回训练数据集和测试数据集
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0, ], [1, ])])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader


if __name__ == '__main__':
    mnist_model = MnistNet('CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = data_loader()
    mnist_model.train(train_loader)
    mnist_model.evaluate(test_loader)
