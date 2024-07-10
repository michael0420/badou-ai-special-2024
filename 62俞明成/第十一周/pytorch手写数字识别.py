import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Model:
    def __init__(self, net, cost, optimizer):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimizer)

    def create_cost(self, cost):
        support_cost = {
            'MSE': torch.nn.MSELoss(),
            'CrossEntropy': torch.nn.CrossEntropyLoss()
        }
        return support_cost[cost]

    def create_optimizer(self, optimizer, **rests):
        support_optimizer = {
            'SGD': torch.optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'Adam': torch.optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': torch.optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optimizer[optimizer]

    def train(self, train_loader, epochs=3):
        for epoch in range(epochs):
            running_loss = 0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # 统计损失,用于后续的平均损失计算 running_loss / 100
                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print("训练完成")

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0

        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = self.net(inputs)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


class MNIST(torch.nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def mnist_load_data():
    # 数据预处理转换
    # 创建一个transform对象
    # 将MNIST数据集中的图像从PIL Image格式转换为PyTorch张量
    # 对转换后的张量进行归一化，将像素值从0-255范围标准化到0-1之间
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0, ], [1, ])
        ]

    )
    train_set = torchvision.datasets.MNIST(root='/data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.MNIST(root='/data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2)
    return train_loader, test_loader


if __name__ == '__main__':
    net = MNIST()
    model = Model(net, 'CrossEntropy', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
