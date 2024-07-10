from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

def mnist_load_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0,), (1,))])
    trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    return trainloader, testloader

if __name__ == '__main__':
    trainloader, testloader = mnist_load_data()
    for i, data in enumerate(trainloader, 0):
        print(data[0][0][0][0][0].shape)
