import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

class CIFARNet(nn.Module):
    def __init__(self):
        super(CIFARNet, self).__init__()
        self.net_list = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding='same', bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding='same', bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(1600, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 10)
        )

    def forward(self, x):
        return self.net_list(x)

class Model:
    def __init__(self, net, loss_fc, optimizer, device, epochs = 5):
        self.net = net
        self.loss_fc = self.create_loss_fc(loss_fc)
        self.optimizer = self.create_optimizer(optimizer)
        self.device = device
        self.epochs = epochs
    
    def create_loss_fc(self, loss_fc):
        loss_functions = {
            'cross_entropy': nn.CrossEntropyLoss(),
            'mse': nn.MSELoss()
        }

        return loss_functions[loss_fc]

    def create_optimizer(self, optimizer, **rests):
        # rest can be any other parameters
        # in the optimizer, rest can be like momentum, weight_decay, etc.
        optimizers = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.00001, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return optimizers[optimizer]

    def train(self, train_loader):
        for epoch in range(self.epochs):
            running_loss = 0.0 
            tqdm_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}', leave=False)
            for batch, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                y_pred = self.net(x)
                loss = self.loss_fc(y_pred, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                tqdm_bar.set_postfix(loss=running_loss/(batch+1))
                tqdm_bar.update()
            print(f'Epoch {epoch+1}, average loss: {running_loss / len(train_loader):.3f}')
        
        print('Finished Training')
    
    def evaluate(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.net(x)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

def cifar_data_load(batch_size = 64):
    transform_train = transforms.Compose([
        transforms.RandomCrop(24),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.8, contrast=[0.2, 1.8]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(24),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    train_loader, test_loader = cifar_data_load()
    cifar_net = CIFARNet().to(device)
    model = Model(cifar_net, 'cross_entropy', 'ADAM', device, epochs=5)
    model.train(train_loader)
    model.evaluate(test_loader)

