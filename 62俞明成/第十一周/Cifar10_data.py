import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# 加载数据并进行预处理
def read_cifar10(file, train):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomCrop(24),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])
    transform_test = transforms.Compose([
        transforms.Resize((24, 24)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if train:
        dataset = torchvision.datasets.CIFAR10(root=file, train=train, download=True, transform=transform_train)
    else:
        dataset = torchvision.datasets.CIFAR10(root=file, train=train, download=True, transform=transform_test)
    return dataset


def load_data(data_dir, batch_size, train=True):
    if train:
        dataset = read_cifar10(data_dir, train=True)
    else:
        dataset = read_cifar10(data_dir, train=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True if train else False, num_workers=4)
    return data_loader


# 测试函数
if __name__ == "__main__":
    data_dir = 'data'  # 定义数据集存储路径
    batch_size = 100  # 批处理大小
    train_loader = load_data(data_dir, batch_size, train=True)
    test_loader = load_data(data_dir, batch_size, train=False)  # 加载测试集

    # 打印示例数据集的大小
    dataiter = iter(train_loader)
    images, labels = next(dataiter)  # 使用 next() 函数
    print(images.size(), labels.size())

    # 打印测试集的一个批次
    dataiter_test = iter(test_loader)
    images_test, labels_test = next(dataiter_test)
    print(images_test.size(), labels_test.size())
