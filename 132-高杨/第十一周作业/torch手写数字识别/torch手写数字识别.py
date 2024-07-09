import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class Model:
    def __init__(self,net,cost,optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimist = self.create_optimizer(optimist)
    def create_cost(self,cost):
        support_cost = {
            'CROSS_ENTROPY':nn.CrossEntropyLoss(),
            'MSE':nn.MSELoss()
        }
        return support_cost[cost]
    def create_optimizer(self,optimist,**rests):
        # 不同优化器 会需要不同的学习率
        support_optim = {
            'SGD':optim.SGD(self.net.parameters(),lr=0.1,**rests),
            'ADAM':optim.Adam(self.net.parameters(),lr=0.01,**rests),
            'RMSP':optim.RMSprop(self.net.parameters(),lr=0.001,**rests)
        }
        return  support_optim[optimist]


    def train(self,train_loader,epoches=3):
        for e in range(epoches):
            runing_loss = 0.0
            # i 是下标，表示第几个
            for i, data in enumerate(train_loader,0):
                inputs,labels =data
                print(f'inputs: {inputs},labels:{labels}')
                self.optimist.zero_grad()
                outputs = self.net(inputs)
                loss = self.cost(outputs,labels)
                loss.backward()
                self.optimist.step()
                runing_loss += loss.item()

                if i%100==0:
                    print(f'epoch:{e+1}, total_accomplish: {(i+1)*1. / len(train_loader)} , running_loss: {runing_loss / 100}',)
                    runing_loss=0.0
    def test(self,test_loader):
        corrt_num = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs,labels = data
                outputs = self.net(inputs)
                predict = torch.argmax(outputs)
                total += labels.size(0)
                corrt_num += (predict == labels).sum().item()

            print('Accuracy of the network on the test images: %d %%' % (100 * corrt_num / total))

def mnist_load_data():
        # 第一步构建transform 利用trochvison把数据转化为张量
        transform = torchvision.transforms.Compose([transforms.ToTensor(),transforms.Normalize(0,1)])
        train_data = torchvision.datasets.MNIST('./mdata',train=True,download=True,transform=transform)

        train_loader = torch.utils.data.DataLoader(train_data,batch_size=32,shuffle=True,num_workers=2)

        test_data = torchvision.datasets.MNIST('./mdata',train=True,download=True,transform=transform)

        test_loader = torch.utils.data.DataLoader(test_data,batch_size = 32,shuffle=True,num_workers=2)

        return train_loader,test_loader


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet,self).__init__()
        self.fc1 = torch.nn.Linear(28*28,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,10)

    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return  x

if __name__ == '__main__':
    net = MnistNet()
    model = Model(net,'CROSS_ENTROPY','RMSP')
    train_loader , test_loader = mnist_load_data()
    model.train(train_loader)
    model.test(test_loader)
