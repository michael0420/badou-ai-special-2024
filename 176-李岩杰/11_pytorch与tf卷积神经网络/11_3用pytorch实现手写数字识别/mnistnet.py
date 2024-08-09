import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


# 目标是用mnist数据集，采用pytorch框架实现器。
#函数1加载数据集，函数2定义模型，函数3训练模型，函数4测试模型。
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5, 1)
        self.MaxPool2d1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5, 1)
        self.Maxpool2d2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.MaxPool2d1(x)
        x = F.relu(self.conv2(x))
        x = self.Maxpool2d2(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.softmax(x)
        return x
MNISTNet = Net()

class Model():
    def train(self, train_loader, epochs=3):
        optimizer = optim.Adam(MNISTNet.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = MNISTNet(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                #print('Epoch: %d, Loss: %.3f' % (epoch + 1, loss.item()))
                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0
                #if i % 100 == 0:
                   # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
                   # print('Finished Training')


    def test(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = MNISTNet(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
    return train_loader, test_loader
if __name__ == '__main__':
    train_loader, test_loader = mnist_load_data()
    mnistnet = Model()
    mnistnet.train(train_loader)
    mnistnet.test(test_loader)






