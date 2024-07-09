# coding = utf-8

'''
    用pytorch实现训练
'''


import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as func
import torchvision
import torchvision.transforms as trans


class Model:
    def __init__(self, net, loss, optimist):
        # 初始化网络、损失函数、优化项
        self.net = net
        self.loss = self.create_loss(loss)
        self.optimist = self.create_opt(optimist)


    def create_loss(self, loss):
        # 根据传入的值确定损失函数
        loss_choice = {
                        'CROSS_ENTROPY': nn.CrossEntropyLoss(),
                        'MSE': nn.MSELoss()
                       }
        return loss_choice[loss]


    def create_opt(self, optimist, **rests):
        # 根据传入的值确定优化操作
        opt_choice = {
                       'SGD': opt.SGD(self.net.parameters(), lr=0.1, **rests),
                       'ADAM': opt.Adam(self.net.parameters(), lr=0.01, **rests),
                       'RMSP': opt.RMSprop(self.net.parameters(), lr=0.001, **rests)
                     }
        return opt_choice[optimist]


    def train(self, train_data, epoch=3):       # epoch默认3
        for i in range(epoch):
            new_loss = 0.0
            for j, data in enumerate(train_data, 0):
                inputs, labels = data
                self.optimist.zero_grad()
                # 正向传播
                outputs = self.net(inputs)
                # 反向传播
                loss = self.loss(outputs, labels)
                loss.backward()
                # 优化操作
                self.optimist.step()
                new_loss += loss.item()
                # 每更新100次输出日志
                if j % 100 == 0:
                    print('[epoch %d, %.4f%%] loss: %.4f'
                          % (i+1, (j+1)*1./len(train_data),
                            new_loss / 100))
                    new_loss = 0.0
            print('='*40, '迭代 %d 次完成' % (i+1), '='*40)
        print('='*40, '训练结束', '='*40)


    def predict(self, test_data):
        correct = 0
        all_data = 0
        # 推理不需要求导
        with torch.no_grad():
            for data in test_data:
                inputs, labels = data
                outputs = self.net(inputs)
                #   argmax(outputs, 1)取结果中最有可能的第一位
                result = torch.argmax(outputs, 1)
                all_data += labels.size(0)
                correct += (result == labels).sum().item()
        print('推理准确率：%d %%' % (100 * correct / all_data))
        print('='*40, '推理结束', '='*40)


def Mnist_Load_Data():
        transform = trans.Compose(
            [trans.ToTensor(),
            trans.Normalize([0, ], [1, ])]
        )
        train_mnist = torchvision.datasets.MNIST(root='./data',
                                   train=True,
                                   download=True,
                                   transform=transform)
        train_loader = torch.utils.data.DataLoader(train_mnist,
                                    batch_size=32,
                                    shuffle=True,
                                    num_workers=2)
        test_mnist = torchvision.datasets.MNIST(root='./data',
                                                train=False,
                                                download=True,
                                                transform=transform)
        test_loader = torch.utils.data.DataLoader(test_mnist,
                                                  batch_size=32,
                                                  shuffle=True,
                                                  num_workers=2)
        return train_loader, test_loader


class MnistNet(torch.nn.Module):
    def __init__(self):
        # 需要训练的层写在init中，输入与上一输出对应
        super(MnistNet, self).__init__()
        # 输入层
        self.fc1 = torch.nn.Linear(28*28, 512)
        # 中间层
        self.fc2 = torch.nn.Linear(512, 512)
        # 输出层
        self.fc3 = torch.nn.Linear(512, 10)


    def forward(self, x):
        # 不需要训练的层写在forward中，更新得到
        # 转换为一维数据输入
        x = x.view(-1, 28*28)
        x = func.relu(self.fc1(x))
        x = func.relu((self.fc2(x)))
        x = func.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    # 定义网络
    net = MnistNet()
    # 构建网络结构
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    # 载入数据
    train_data, test_data = Mnist_Load_Data()
    # 开始训练
    epoch = 2
    model.train(train_data, epoch)
    # 推理
    model.predict(test_data)
