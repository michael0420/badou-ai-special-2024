import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# 步骤3: 加载预训练的FasterR - CNN模型
# 加载预训练的Faster R-CNN模型
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # 设置为评估模式

# 如果需要GPU支持，可以移动模型到GPU
if torch.cuda.is_available():
    model = model.cuda()

# 步骤4: 图像预处理
# 在将图像输入模型之前，需要对其进行预处理以匹配模型训练时使用的格式。

def get_transform():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # 将PIL图像或numpy.ndarray转换为tensor
    ])
    return transforms


# 假设 image 是PIL图像或numpy数组
image = ...  # 加载你的图像
transform = get_transform()
image_tensor = transform(image).unsqueeze(0)  # 增加一个batch维度

if torch.cuda.is_available():
    image_tensor = image_tensor.cuda()

# 步骤5: 进行预测
with torch.no_grad():
    prediction = model(image_tensor)

# prediction 包含boxes, labels, scores
print(prediction)
