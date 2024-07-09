import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms


class Model:
    def __init__(self, net, loss, optimist):
        self.net = net
        self.loss_f = self.parse_loss(loss)
        self.optimizer = self.parse_optimizer(optimist)

    def parse_loss(self, loss):
        loss_holder = {
            "CROSS_ENTROPY": nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss(),
        }
        return loss_holder[loss]

    def parse_optimizer(self, optimist, **rests):
        optimizer_holder = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return optimizer_holder[optimist]

    def train(self, train_loader, epoches=3):
        for _epoch in range(epoches):
            running_loss = 0.0
            for i, item in enumerate(train_loader):
                data, label = item
                self.optimizer.zero_grad()

                outputs = self.net(input)
                loss = self.loss_f(outputs, label)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (_epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for test in test_loader:
                test_data, test_label = test

                outputs = self.net(test_data)
                argmax = torch.argmax(outputs, 1)
                total += test_label.size(0)
                correct += (argmax == test_label).sum().item()
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


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    # train for mnist
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
