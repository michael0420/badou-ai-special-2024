import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class small_model:
    def __init__(self, netWork, loss_function, optimization_items):
        self.netWork = netWork
        self.loss_function = self.create_cost(loss_function)
        self.optimization_items = self.create_optimizer(optimization_items)
        pass

    # 选择使用什么
    def create_cost(self, loss_function):
        if "CROSS_ENTROPY" == loss_function:
            return nn.CrossEntropyLoss()
        if "MSE" == loss_function:
            return nn.MSELoss()

    def create_optimizer(self, optimist, **rests):
        # 定义字典
        my_optimist = dict()
        my_optimist['SGD'] = optim.SGD(self.net.parameters(), lr=0.1, **rests)
        my_optimist['ADAM'] = optim.Adam(self.net.parameters(), lr=0.01, **rests)
        my_optimist['RMSP'] = optim.Adam(self.net.parameters(), lr=0.001, **rests)
        return my_optimist.get(optimist)

    def train(self, data, count=3):
        for epch in count:
            running_loss = 0.0
            for index, record in enumerate(data, 0):
                train_data, lable = record
                self.optimization_items.zero_grad()
                work = self.netWork(train_data)
                loss = self.loss_function(work, lable)
                loss.backward()
                self.optimization_items.step()

                running_loss += loss.item()
                if index % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epch + 1, (index + 1) * 1. / len(data), running_loss / 100))
                    running_loss = 0.0
        print('Finished Training')

    def test_and_verify(self, test_data):
        print('Evaluating ...')
        correct = 0
        total = 0
        # 此范围不进行求导
        with torch.no_grad():
            for shuju in test_data:
                test_shuju, test_lable = shuju
                work = self.netWork(test_shuju)
                argmax = torch.argmax(work, 1)
                total += test_lable.size()
                correct += (argmax == test_lable).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


class net_work(torch.nn.Module):
    def __init__(self):
        super(net_work, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 618)
        self.fc3 = torch.nn.Linear(618, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def loading_data():
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


if __name__ == '__name__':
    net = net_work()
    model = small_model(net, 'CROSS_ENTROPY', 'RMSP')
    train_data, test_data = loading_data()
    model.train(train_data)
