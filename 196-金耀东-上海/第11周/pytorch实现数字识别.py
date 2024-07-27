from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.nn import Module
import torch

#定义超参数
NUM_PIXELS = 28 * 28 # 输入图片像素
NUM_CLASSES = 10 # 标签类别总数0-9
NUM_HIDDE = 16 # 隐藏节点数
NUM_EPOCHS = 3 # 训练代数
BATCH_SIZE = 600 # 批次大小
LR = 1e-2 # 学习率

class NeuralNetwork(Module):
    def __init__(self, in_dims, out_dims, hidde_units):
        super(NeuralNetwork, self).__init__()
        # 定义网络层
        self.fc1 = torch.nn.Linear(in_features=in_dims,out_features=hidde_units)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=hidde_units, out_features=out_dims)

        # 定义损失函数:交叉熵
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        # 定义优化器
        self.optmizer = torch.optim.Adam(params=self.parameters())

    def _foward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def _backward(self, loss):
        self.optmizer.zero_grad()
        loss.backward()
        self.optmizer.step()

    def train(self, dataset_loard, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            for imgs, labels in dataset_loard:
                # 输入->正向传播
                y_hat = self._foward( imgs.view([-1, NUM_PIXELS]) )

                # 计算损失
                loss = self.cross_entropy_loss(y_hat, labels)

                # 损失->反向传播
                self._backward(loss)

            # 评估当前训练效果
            loss_train , accuracy_train = self.evalute(dataset_loard)
            print(f"training [ epoch:{epoch+1} -- loss:{loss_train} -- accuracy:{accuracy_train} ]")

    def evalute(self, dataset_loard):
        # 评估模型：返回损失（loss）和准确率（accuracy）
        with torch.no_grad():# 暂时禁用自动微分
            num_total , num_correct , loss_total = 0 , 0 , 0.0
            for imgs, labels in dataset_loard:
                y_hat = self._foward(imgs.view([-1, NUM_PIXELS]))
                loss_total += self.cross_entropy_loss(y_hat, labels)
                predictions = y_hat.argmax(axis=1)
                num_correct += (predictions == labels).sum().item()
                num_total += labels.size(0)
            return loss_total/num_total , num_correct/num_total

def get_data_loard(is_train, is_shuffle, batch_size):
    # 定义图像预处理顺序：先转化为Tensor,再标准化
    transforms_compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize( mean=(0.1307,), std=(0.3081,) ) # mnist样本均值：0.1307；标准差：0.3081
    ])
    dataset = MNIST(root="./data",train=is_train, transform=transforms_compose, download=True)
    return DataLoader(dataset, batch_size, shuffle=is_shuffle)

if __name__ == "__main__":
    # 获取mnist数据,并生成加载器
    data_loard_train = get_data_loard(is_train=True, is_shuffle=True, batch_size=BATCH_SIZE)
    data_loard_test = get_data_loard(is_train=False, is_shuffle=False, batch_size=BATCH_SIZE)

    # 创建模型
    model = NeuralNetwork(in_dims=NUM_PIXELS, out_dims=NUM_CLASSES, hidde_units=NUM_HIDDE)

    # 训练模型
    model.train(data_loard_train, num_epochs=NUM_EPOCHS, learning_rate=LR)

    # 评估模型
    loss , accuracy = model.evalute(data_loard_test)
    print(f"evalute [ loss:{loss} -- accuracy:{accuracy} ]")
