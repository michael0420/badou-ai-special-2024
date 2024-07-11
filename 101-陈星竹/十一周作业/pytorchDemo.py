#1.pytorch实现识别手写数字
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Model:
    def __init__(self,net,cost,optimist):
        self.net = net
        self.cost = self.create_cost(cost)#损失函数的名称
        self.optimizer = self.create_optimist(optimist) #优化器

    #根据传入的损失函数类型，创建实际的损失函数
    def create_cost(self,cost):
        support_cost = {
            'CROSS_ENTROPY':nn.CrossEntropyLoss(),
            'MSE':nn.MSELoss()
        }
        return support_cost[cost]

    #根据传入的优化器类型，创建实际的优化器
    def create_optimist(self,optimist,**rests):
        support_optim = {
            'SGD':optim.SGD(self.net.parameters(),lr=0.1,**rests),
            'ADAM':optim.Adam(self.net.parameters(),lr=0.01,**rests),
            'RMSP':optim.RMSprop(self.net.parameters(),lr=0.001,**rests)
        }
        return support_optim[optimist]

    #训练部分,默认训练批次为3
    def train(self,train_loader,epoches=3):
        for epoch in range(+epoches):
            running_loss = 0.0 #每一轮开始前都将loss初始化为0
            for i,data in enumerate(train_loader,0):
                inputs,labels = data #将标签和数据分离
                self.optimizer.zero_grad() #梯度清0，清除上一轮的迭代的梯度信息
                outputs = self.net(inputs) #向前传播
                loss = self.cost(outputs,labels) #计算损失
                loss.backward()#反向传播,backwrad求导
                self.optimizer.step()#优化器更新网络参数
                running_loss+=loss.item()

                #每100次打印一次当前的平均损失，
                if i%100==0:
                    print('[epoch %d,%.2f%%  loss:%.3f]'%(epoch+1,(1+i)*1./len(train_loader),running_loss/100))
                    running_loss = 0.0 #重置
        print('Finished Trainning')
    #推理部分
    def evaluate(self,test_loader):
        correct = 0
        total = 0
        with torch.no_grad(): #关闭梯度计算
            for data in test_loader:
                images,labels = data
                outpus = self.net(images)
                predicted = torch.argmax(outpus,1) #返回每个样本概率最大的类别索引
                total+=labels.size(0)
                #predicted == labels 生成一个布尔张量，
                # 其中每个元素表示 predicted 和 labels 是否相等。
                # 然后，.sum() 方法计算这个布尔张量中 True 值的总数
                correct += (predicted==labels).sum().item()
        print("正确率为：%d %%"%(100*correct/total))


#加载数据
def mnist_load_data():
    #ToTensor()将shape为(H,W,C)的PIL图像或者Numpy数组转换成shape为(C,H,W)的tensor，归一化到[0,1]
    #transforms.Normalize(mean, std)：对张量进行归一化处理。这里的 mean 和 std 分别是要使用的均值和标准差。
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,],[1,])])
    #root:数据集存放的位置
    #transform（可调用，可选）–接受PIL图像并返回已转换版本的函数/转换。
    trainset = torchvision.datasets.MNIST(root="./data",train=True,
                                          download=True,transform=transform)
    #使用 DataLoader 将训练数据集包装为可迭代对象，设置批量大小为32，数据打乱，使用2个工作线程。
    '''
    将训练数据集包装为 DataLoader 对象，并配置适当的参数，可以提高训练效率，
    增加模型的泛化能力，并更好地利用硬件资源。这些设置使得训练过程更加高效和稳定。
    '''
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,
                                              shuffle=True,num_workers=2)
    #加载测试数据集
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    #测试数据加载器
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)
    return trainloader,testloader

#定义神经网络
class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28,512)
        self.fc2 = torch.nn.Linear(512,512)
        self.fc3 = torch.nn.Linear(512,10) #10个类别 10个数字

     #前向传播
    def forward(self,x):
        x = x.view(-1,28*28) #reshape -1表示维度不变
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

if __name__=='__main__' :
    net = MnistNet()
    model = Model(net,'CROSS_ENTROPY','RMSP')
    trainLoader,testLoader = mnist_load_data()
    model.train(trainLoader)
    model.evaluate(testLoader)

