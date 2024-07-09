'''
使用pytorch实现简单的NeuralNetwork
'''

import torch
import torch.nn.functional as F
import torchvision as tv


class Model():
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': torch.nn.CrossEntropyLoss(),
            'MSE': torch.nn.MSELoss()
        }
        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': torch.optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': torch.optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': torch.optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]

    def train(self,train_loader, epoch=3):
        for i in range(epoch):
            running_loss = 0.0
            for index, data in enumerate(train_loader):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if index % 100 == 0:
                    print(f'【epoch:{i+1},{(index + 1)*1./len(train_loader)}】,loss:{running_loss / 100}')
                    running_loss = 0.0

    def evaluate(self, test_loader):
        print('evaluating...................')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data

                outputs = self.net(inputs)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'准确率：{100 * correct / total}')

        pass


def mnist_load_data():
    transform = tv.transforms.Compose(
        [tv.transforms.ToTensor(),
         tv.transforms.Normalize([0, ], [1, ])]
    )
    trainset = tv.datasets.MNIST(root='./data', train=True,
                                 download=True, transform=transform)
    trainlaoder = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = tv.datasets.MNIST(root='./data', train=False,
                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=True, num_workers=2)
    return trainlaoder, testloader



class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
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