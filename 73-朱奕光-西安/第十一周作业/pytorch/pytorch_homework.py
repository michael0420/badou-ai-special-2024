import torch
import torch.nn.functional as F
from pytorch_load_data_homework import mnist_load_data

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28,300)
        self.fc2 = torch.nn.Linear(300,200)
        self.fc3 = torch.nn.Linear(200,10)

    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x

class Model(object):
    def __init__(self,net,cost,optimizer):
        self.net = net
        self.cost = cost
        self.optimizer = optimizer

    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            for i, data in enumerate(train_loader, 0):
                img, labels = data
                self.optimizer.zero_grad()
                output = self.net(img)
                loss = self.cost(output,labels)
                loss.backward()
                self.optimizer.step()
                running_loss = loss.item()
                if i % 100 == 0:
                    print(f'第{epoch+1}次epochs，进度{round((i+1)*1./len(train_loader)*100,2)}%,loss为{round(running_loss/100,4)}')
                    running_loss = 0.0
        print('训练结束')

    def eval(self,test_loader):
        print('正在验证')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                img, labels = data
                output = self.net(img)
                predicted = torch.argmax(output,dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'准确率为{round(correct/total*100,2)}%')


if __name__ == '__main__':
    net = Net()
    model = Model(net, cost=torch.nn.CrossEntropyLoss(), optimizer=torch.optim.RMSprop(net.parameters(),lr=0.001))
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader, epochs=3)
    model.eval(test_loader)