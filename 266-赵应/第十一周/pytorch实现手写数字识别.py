import torch
import torchvision
import torchvision.transforms as transforms


def mnist_load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0, ], [1, ])])
    train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    test_data = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=2)
    return train_loader, test_loader


class Model:
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
        support_optimist = {
            'SGD': torch.optim.SGD(self.net.parameters(), lr=.1, **rests),
            'ADAM': torch.optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': torch.optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optimist[optimist]

    def train(self, train_loader, epochs=3):
        for epoch in range(epochs):
            running_loss = .0
            for i, data in enumerate(train_loader, 0):
                train_data, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(train_data)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if (i + 1) % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating...')
        correct = 0
        total = 0
        with total.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(521, 10)

    def forward(self, inputs):
        outputs = inputs.view(-1, 28, 28)
        outputs = torch.nn.functional.relu(self.fc1(outputs))
        outputs = torch.nn.functional.relu(self.fc2(outputs))
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        return outputs


if __name__ == '__main__':
    train_data_loader, test_data_loader = mnist_load_data()
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    model.train(train_data_loader)
    model.evaluate(test_data_loader)
