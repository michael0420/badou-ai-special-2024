# 导入必要的库

import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义深度可分离卷积

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = nn.functional.relu6(out)  # MobileNet 使用 ReLU6 激活函数
        return out


# 定义MobileNetV1架构



class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1, padding=1),
            DepthwiseSeparableConv(64, 128, kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1, padding=1),

            # 这里可以继续添加更多的层，根据 MobileNetV1 的具体架构
            # 注意：MobileNetV1 有多个深度可分离卷积块，并带有 stride=2 的层来减少空间维度

            # 示例结束，实际应添加更多层
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 1024),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 使用
# MobileNetV1

model = MobileNetV1(num_classes=1000)  # 假设有 1000 个类别
print(model)

# 假设有一个输入 tensor
input_tensor = torch.randn(1, 3, 224, 224)  # batch_size=1, channels=3, height=224, width=224
output = model(input_tensor)
print(output.shape)  # 应该输出 torch.Size([1, 1000])
