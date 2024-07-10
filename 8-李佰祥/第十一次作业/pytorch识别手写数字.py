import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


#使用pytorch的方式读取测试集和训练集数据
##transform允许你在加载数据时立即对数据进行预处理
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0,), (1,))])

trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True,num_workers=2)

testset = torchvision.datasets.MNIST(root='./data',train=False,download = True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=32,shuffle=True,num_workers=2)

#构建网络结构
class MNISTnet(torch.nn.Module):
    #在init里面定义层
    def __init__(self):
        super(MNISTnet, self).__init__()  #如果不调用父类的init方法（父类的init中会包含参数管理、训练/评估模式切换、递归遍历子模块），会导致无法进行模型构建
        self.fc1 = torch.nn.Linear(28*28,512)  #这样写得到的fc1是torch.nn.Linear的实例
        self.fc2 = torch.nn.Linear(512,512)
        self.fc3 = torch.nn.Linear(512,10)
    #在forward（正向传播）里面定义数据shape和激活函数
    def forward(self,x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1) #softmax函数将数值转换为总和为1的概率分布，dim=1就是按照列维度计算softmax
        return x

#定义模型，其中包含了参数初始化，训练和评估模型的方法
class Model:
    #初始化网络结构，损失函数，优化器
    def __init__(self,net,cost,optimizer):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimizer)

    #定义支持的损失函数
    def create_cost(self,cost):
        support_cost = {
            'CROSS_ENTROPY':nn.CrossEntropyLoss(),
            'MSE':nn.MSELoss()
        }
        return support_cost[cost]

    #定义支持的优化器
    def create_optimizer(self,optimizer):
        # 需要知道这三个解决了什么问题，SGD解决了传统批量梯度下降在计算大数据集时的高成本问题
        # 使用小批量数据进行更新。
        # RMSP解决了SGD收敛过慢的问题
        # ADAM，收敛速度快（很快找到损失函数最小值）
        # **rests允许用户在创建优化器时传递任意数量的附加参数，这些参数将直接传递给优化器的构造函数
        # 例如：optimizer = create_optimizer('ADAM', weight_decay=0.001)，
        # eight_decay参数将会被收集到rests字典中，并在optim.Adam构造函数中使用，最终影响优化器的配置
        support_optimizer={
            'SGD':optim.SGD(self.net.parameters(),lr=0.1),
            'ADAM':optim.Adam(self.net.parameters(),lr=0.01),
            #parameters()返回一个迭代器，这个迭代器包含了所有层的参数张量
            'RMSP':optim.RMSprop(self.net.parameters(),lr=0.001)
        }
        return support_optimizer[optimizer]

    #训练函数
    def train(self,trainloader,epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()#将当前优化器中所有参数的梯度清零，如果不及时清零，梯度会累积，导致后续的优化步骤受到之前步骤的影响

                # forward + backward + optimize
                #这个net是一个继承了torch.nn.Module的模型实例，当直接实例传入参数时
                #PyTorch会自动调用该模型实例的forward()方法，并将inputs作为参数传入
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step() #反向传播结束后更新模型参数

                running_loss += loss.item()  #.item()直接获取loss张量中的数值
                if i % 100 == 0:   #每一百次打印一次loss
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(trainloader), running_loss / 100))
                    running_loss = 0.0


    #评估函数
    def evaluate(self,testloader):
        correct = 0
        total = 0
        with torch.no_grad():   # with代码块以下不需要计算梯度，关闭自动求导功能，避免不必要的计算和内存消耗
            for data in testloader:
                images, labels = data

                output = self.net(images)
                #print(output)
                #print("--------")
                predict = torch.argmax(output,1)
                #print(predict)
                total += labels.size(0)
                #print("---------")
                #print("lables: ",labels)
                #print("-----------")
                correct += (predict==labels).sum().item()
                print('Accuracy of each batch  network on the test images: %d %%' % (100 * correct / total))

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))





















if __name__ == '__main__':
    net = MNISTnet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    model.train(trainloader)
    model.evaluate(testloader)






