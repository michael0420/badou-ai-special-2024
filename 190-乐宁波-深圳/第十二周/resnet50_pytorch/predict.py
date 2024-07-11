import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

from ResNet9 import ResNet9

# 定义图片预处理
transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


# 定义预测函数
def predict_image(image_path, model, transform, device):
    # 打开图片
    image = Image.open(image_path).convert('RGB')
    # 预处理图片
    preprocessed_image = transform(image)
    # 增加batch维度
    image_batch = preprocessed_image.unsqueeze(0)
    # 将图片移到GPU上
    image_batch = image_batch.to(device)
    # 设置模型为评估模式
    model.eval()
    with torch.no_grad():
        # 进行前向传播
        output = model(image_batch)
        # 获取预测结果
        _, predicted = torch.max(output, 1)
    return predicted.item(), preprocessed_image


inv_normalize = transforms.Normalize(
    mean=[-0.5071 / 0.2675, -0.4867 / 0.2565, -0.4408 / 0.2761],
    std=[1 / 0.2675, 1 / 0.2565, 1 / 0.2761]
)

# 加载训练好的模型
pretrained_model_path = 'resnet9_cifar100.pth'
model = ResNet9(3, num_classes=100)
if os.path.exists(pretrained_model_path):
    print("Loading pretrained model...")
    model.load_state_dict(torch.load(pretrained_model_path))
else:
    print("Pretrained model not found.")

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 预测自定义图片
image_folder = 'predict_data'
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('png', 'jpg', 'jpeg'))]

for image_path in image_paths:
    predicted_class_idx, preprocessed_image = predict_image(image_path, model, transform_test, device)
    predicted_class_name = cifar100_classes[predicted_class_idx]

    # 反归一化
    image_for_display = inv_normalize(preprocessed_image)
    # 转换为numpy数组
    image_for_display = image_for_display.permute(1, 2, 0).cpu().numpy()

    # 显示图片及其预测结果
    plt.imshow(image_for_display)
    plt.title(f'Predicted: {predicted_class_name}')
    plt.axis('off')
    plt.show()
