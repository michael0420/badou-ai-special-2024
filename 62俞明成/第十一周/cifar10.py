import torch
from torch import nn
from torch.nn import functional as F
from Cifar10_data import load_data


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Padding = ((Stride - 1) * Input_size + Kernel_size - Stride) / 2
        # padding = (Kernel_size - Stride) / 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(64 * 6 * 6, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        # 卷积池化ReLu
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        # 第二个卷积层
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        # 扁平化
        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        # 去除第一维，即batch_size
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features


def train(model, train_loader, optimizer, criterion, epochs=5):
    # model.image()
    for epoch in range(epochs):
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)

            l2_lambda = 0.001
            l2_reg = torch.tensor(0.)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"epoch:{epoch},iteration:{i + 1},loss:{running_loss / 100:.3f}")
                running_loss = 0
    # torch.save(model.state_dict(), "cifar10.pth")
    print("训练完成")


def test(model, test_loader):
    # model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            output = model(inputs)
            predicted = torch.argmax(output, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if __name__ == '__main__':
    data_dir = './data'  # 定义数据集存储路径
    batch_size = 32  # 批处理大小
    train_loader = load_data(data_dir, batch_size, train=True)
    test_loader = load_data(data_dir, batch_size, train=False)
    train(model, train_loader, optimizer, criterion, epochs=7)
    test(model, test_loader)
