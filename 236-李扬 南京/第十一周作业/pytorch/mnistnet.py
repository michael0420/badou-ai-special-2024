import torch
import torch.nn as nn
# import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

def mnist_load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0,],[1,])])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader

class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    def create_cost(self, cost):
        supportcost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return supportcost[cost]

    def create_optimizer(self, optimist, **rests):
        supportcost = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests),
        }
        return supportcost[optimist]

    def train(self, train_loder, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loder, 0):
                inputs, labels = data
                # print('-----------')
                # print(labels)
                # print('-----------')

                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %(epoch + 1, (i + 1)*1./len(train_loder), running_loss/100))
                    running_loss = 0.0

        print('finish train')


    def evaluate(self, test_loader):
        print('Evaluate...')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                # print('+++++++++++')
                # print(labels)
                # print('+++++++++++')
                outputs = self.net(images)
                print(outputs)
                predicted = torch.argmax(outputs, 1)
                print(predicted)
                total += labels.size(0)
                print(labels)
                correct += (predicted == labels).sum().item()

        print('accuraacy: %d %%' %(100 * correct / total))

if __name__ == '__main__':
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)

