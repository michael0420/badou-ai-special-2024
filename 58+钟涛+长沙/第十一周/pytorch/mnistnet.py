import torchvision.transforms as transforms
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class Model:
    def __init__(self, net,cost,optimizer):
        self.net = net
        self.cost = cost
        self.optimizer = optimizer

    def create_cost(self, cost):
        support_cost = {
              "Cross_Entropy_Loss":nn.CrossEntropyLoss(),
              "MSE": nn.MSELoss()
                        }
        return support_cost[cost]

    def create_optimizer(self, optimizer, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimizer]


    def train(self,train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i,data in enumerate(train_loader, 0):
                inputs, labels = data
                #梯度清零
                self.optimizer.zero_grad()
                #调用网络模型
                outputs = self.net(inputs)
                #计算损失函数
                loss = self.cost(outputs, labels)
                #反向传播
                loss.backward()
                # 修改值
                self.optimizer.step()

                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def test(self,test_loader):
        print("start test")
        correct = 0
        total = 0
        #不进行梯度更新
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = self.net(inputs)
                predicted = torch.argmax(outputs, -1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28,512)
        self.fc2 = torch.nn.Linear(512,512)
        self.fc3 = torch.nnl.Linear(512, 10)

    def forword(self, x):
        #转为一维
        x = x.view(-1,28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim = 1)
        return x


def data_load():
    transfrom = transforms.Compose([
                                     transforms.ToTensor(),  # 张量
                                    transforms.Normalize([0,],[1,])]) #归一化

    #下载训练数据
    trainset = torchvision.datasets.MNIST(transform=transfrom, download = True,root = './data',train=True)

    #下载数据，多线程=2 ，打乱，
    trainloader = torch.utils.data.DataLoader(trainset,batch_size = 32, shuffle = True, num_workers =2 )

    #下载测试数据
    testset = torchvision.datasets.MNIST(transform=transfrom, download=True, root='./data', train=False)

    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    return trainloader, testloader

if __name__ == '__main__':
    net = MnistNet()
    modle = Model(net, "Cross_Entropy_Loss", "RMSP")
    # 加载数据
    trainloader, testloader = data_load()
    modle.train(trainloader)
    modle.test(testloader)



