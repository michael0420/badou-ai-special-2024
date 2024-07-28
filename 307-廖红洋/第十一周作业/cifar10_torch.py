# -*- coding: utf-8 -*-
"""
@File    :   cifar-10_torch.py
@Time    :   2024/07/05 17:10:27
@Author  :   廖红洋 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import io
import sys
import os

# 定义超参数
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 50

# 数据准备
transform_trian = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),  # 灰度版本为三通道
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 转化为浮点
    ]
)
train_data = datasets.CIFAR10(
    root="Cifar/cifar_data/cifar-10-batches-bin/cifar-10-python.tar",
    train=True,
    transform=transform_trian,
    download=False,
)
test_data = datasets.CIFAR10(
    root="Cifar/cifar_data/cifar-10-batches-bin/cifar-10-python.tar",
    train=False,
    transform=transform_test,
    download=False,
)
train_loader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
test_loader = DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)


# 定义网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Flatten(),  # 数据扁平化，即全连接层，将输入变为一维进行输出
            torch.nn.Linear(in_features=8192, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=10),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, input):
        output = self.model(input)
        return output


model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mode = model.to(device)

if __name__ == "__main__":
    # 模型训练
    writer = SummaryWriter("cifar-10")
    for epoch in range(1, EPOCHS + 1):
        for idx, data in enumerate(train_loader):
            # 解构，送入GPU
            input, label = data
            input, label = input.to(device), label.to(device)
            # 运行模型
            output = model(input)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 可视化
            if idx % 50 == 0:
                writer.add_scalar(
                    "Train/Loss", loss.item(), epoch * len(train_loader) + idx
                )
        print("epoch{} loss:{:.4f}".format(epoch + 1, loss.item()))

    # 保存模型参数
    torch.save(model, "torch_cifar.pt")
    # 模型加载
    model = torch.load("torch_cifar.pt")
    # 测试
    # model.eval()
    model.train()

    correct, total = 0, 0
    for j, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 前向传播
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
        # 准确率可视化
        if j % 20 == 0:
            writer.add_scalar("Train/Accuracy", 100.0 * correct / total, j)

    print("准确率：{:.4f}%".format(100.0 * correct / total))
