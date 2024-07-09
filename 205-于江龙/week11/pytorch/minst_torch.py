import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

class MINSTNet(nn.Module):
    def __init__(self):
        super(MINSTNet, self).__init__()
        self.net_list = torch.nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.net_list(x)

class Model:
    def __init__(self, net, loss_fc, optimizer, device):
        self.net = net
        self.loss_fc = self.create_loss_fc(loss_fc)
        self.optimizer = self.create_optimizer(optimizer)
        self.device = device
    
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
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return optimizers[optimizer]
    
    def train(self, train_loader, epochs = 5):
        for epoch in range(epochs):
            running_loss = 0.0
            tqdm_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
            for batch, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                y_pred = self.net(x)
                loss = self.loss_fc(y_pred, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                tqdm_bar.set_postfix(loss=running_loss, avg_loss=running_loss / (batch + 1))

            print(f'Epoch {epoch+1}, average loss: {running_loss / len(train_loader):.3f}')
        
        print('Finished Training')
    
    def evaluate(self, test_loader):
        print('Starting Evaluation')
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.net(x)
                predicted = torch.argmax(y_pred, dim=1)
                total += y.size(0) # same as len(y)
                correct += (predicted == y).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.3f}%')
    
def minst_load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])

    train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=32, shuffle=True)

    test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test, batch_size=32, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    train_loader, test_loader = minst_load_data()
    net = MINSTNet().to(device)
    model = Model(net, 'cross_entropy', 'RMSP', device)
    model.train(train_loader)
    model.evaluate(test_loader)