import torch
import torch.nn as nn

class MobleNetV1(nn.Module):
    def __init__(self, num_classes):
        super(MobleNetV1, self).__init__()
        self.conv1 = self._conv_st(3, 32, 2)
        self.conv_dw1 = self._conv_dw(32, 64, 1)
        self.conv_dw2 = self._conv_dw(64, 128, 2)
        self.conv_dw3 = self._conv_dw(128, 128, 1)
        self.conv_dw4 = self._conv_dw(128, 256, 2)
        self.conv_dw5 = self._conv_dw(256, 256, 1)
        self.conv_dw6 = self._conv_dw(256, 512, 2)
        self.conv_dw_x5 = self._conv_x5(512, 512, 5)
        self.conv_dw7 = self._conv_dw(512, 1024, 2)
        self.conv_dw8 = self._conv_dw(1024, 1024, 1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_dw1(x)
        x = self.conv_dw2(x)
        x = self.conv_dw3(x)
        x = self.conv_dw4(x)
        x = self.conv_dw5(x)
        x = self.conv_dw6(x)
        x = self.conv_dw_x5(x)
        x = self.conv_dw7(x)
        x = self.conv_dw8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        y = torch.softmax(x)
        return x, y

    def _conv_x5(self, in_channel, out_channel, blocks):
        layers = []
        for i in range(blocks):
            layers.append(self._conv_dw(in_channel, out_channel, 1))
        return nn.Sequential(*layers)

    def _conv_st(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def _conv_dw(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
net = MobleNetV1(1000)
x = torch.rand(1,3,224,224)
for name,layer in net.named_children():
    if name != "fc":
        x = layer(x)
        print(name, 'output shape:', x.shape)
    else:
        x = x.view(x.size(0), -1)
        x = layer(x)
        print(name, 'output shape:', x.shape)

