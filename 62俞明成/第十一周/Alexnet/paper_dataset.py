import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

# 假设我们有一个LabelEncoder实例来转换标签到整数
# label_encoder = LabelEncoder()


class ImageDataset(Dataset):
    def __init__(self, lines, transform=None, num_classes=2, train=True):
        self.lines = lines
        self.train = train
        self.transform = transform
        self.num_classes = num_classes
        self.labels = [line.split(';')[1] for line in lines]
        # self.label_encoder = label_encoder.fit(self.labels)
        # 打乱lines列表，因为Dataset通常会在每个epoch开始时打乱数据
        np.random.shuffle(self.lines)

    def __len__(self):
        return len(self.lines)  # 返回数据集的总长度

    def __getitem__(self, idx):
        name = self.lines[idx].split(';')[0]
        img_path = r"./data/image" + '/' + name
        img = cv2.imread(img_path)
        if img is None:
            # 图像可能不存在或路径错误，这里需要处理这种情况
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)  # 应用图像转换（例如resize）
        label = int(self.lines[idx].split(';')[1])
        # label = self.label_encoder.transform([self.lines[idx].split(';')[1]])[0]

        # 将图像和标签转换为Tensor
        img_tensor = torch.tensor(img).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

batch_size = 128
with open('data/dataset.txt', 'r') as f:
    lines = f.readlines()

# 分离训练集和测试集
num_train = int(len(lines) * 0.8)
train_lines = lines[:num_train]
test_lines = lines[num_train:]

dataset_train = ImageDataset(train_lines, transform=transform, num_classes=2, train=True)
dataset_test = ImageDataset(test_lines, transform=transform, num_classes=2, train=False)

# 使用DataLoader来加载数据
data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

if __name__ == '__main__':
    pass
    dataiter = iter(data_loader_train)
    images, labels = next(dataiter)  # 使用 next() 函数
    print(images.size(), labels.size())

    # 印测试集的一个批次
    dataiter_test = iter(data_loader_test)
    images_test, labels_test = next(dataiter_test)
    print(images_test.size(), labels_test.size())
