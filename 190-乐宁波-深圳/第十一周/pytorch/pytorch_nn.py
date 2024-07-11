import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class MyNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyNeuralNet, self).__init__()

        self.hidden_layer1 = nn.Linear(input_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)# 这一行是什么
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x


def mnist_load_data():
    # 创建一个转换序列，并将其缩放到均值为0，标准差为1的范围内
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0, ], [1, ])])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True, num_workers=2)
    return train_loader, test_loader


if __name__ == '__main__':
    input_size = 28 * 28
    hidden_size = 28 * 28
    num_classes = 11
    num_epochs = 1
    model = MyNeuralNet(input_size, hidden_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader, test_loader = mnist_load_data()

    model.train()
    for epoch in range(num_epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print('[epoch %d] loss: %.3f' %
                  (epoch + 1, loss))

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            predictions = output.argmax(dim=1)
            total += target.size(0)
            correct += (predictions == target).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    torch.save(model.state_dict(), 'model_weights.pth')
    model.load_state_dict(torch.load('model_weights.pth'))
