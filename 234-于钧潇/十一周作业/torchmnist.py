import torch
import torch.nn as nn
import torch.nn.functional as fc
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.ly1 = nn.Linear(28*28, 512)
        self.ly2 = nn.Linear(512, 1024)
        self.ly3 = nn.Linear(1024, 10)

    def forward(self, input):
        tmp = input.view(-1, 28*28)
        tmp = fc.relu(self.ly1(tmp))
        tmp = fc.relu(self.ly2(tmp))
        tmp = fc.softmax(self.ly3(tmp), dim=1)
        return tmp

class Model:
    def __init__(self, net, costname, optimistname):
        self.net = net
        self.costfc = self.create_cost(costname)
        self.optimist = self.create_optimist(optimistname)

    def create_cost(self, name):
        cost = {"CROSS_ENTROPY": nn.CrossEntropyLoss(),
                "MSE": nn.MSELoss()}
        return cost[name]

    def create_optimist(self, name, **rests):
        optimist = {"SGD": optim.SGD(self.net.parameters(), lr=0.1, **rests),
                    "ADAM": optim.Adam(self.net.parameters(), lr=0.01, **rests),
                    "RMSP": optim.RMSprop(self.net.parameters(), lr=0.001, **rests)}
        return optimist[name]

    def train(self, train_data, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_data):
                inputs, label =data
                self.optimist.zero_grad()

                outputs = self.net(inputs)
                loss = self.costfc(outputs, label)
                loss.backward()
                self.optimist.step()
                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] %d %d loss: %.3f' %
                          (epoch + 1, (i + 1)*100./len(train_data), i, len(train_data), running_loss / 100))
                    running_loss = 0.0
        print('Finished Training')

    def evaluate(self, test_data):
        correct = 0
        total = 0
        # 不计算梯度
        with torch.no_grad():
            for data in test_data:
                image, label = data
                output = self.net(image)
                predict = torch.argmax(output, 1)
                total += label.size(0)
                correct += (predict == label).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))





def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
    return trainloader, testloader

if __name__ == '__main__':
    net = MnistNet()
    print(net)
    model = Model(net, 'CROSS_ENTROPY', 'SGD')
    train_loader, test_loader = mnist_load_data()

    dataiter = iter(train_loader)
    # images 四个维度 BCHW
    images, labels = dataiter.next()
    print(images.shape, labels, len(train_loader))


    model.train(train_loader)
    model.evaluate(test_loader)
