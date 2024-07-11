import torch.nn as nn


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 16)
        self.conv2 = conv_block(16, 32, pool=True)
        self.res1 = nn.Sequential(conv_block(32, 32), conv_block(32, 32))

        self.conv3 = conv_block(32, 32, pool=True)
        self.res2 = nn.Sequential(conv_block(32, 32), conv_block(32, 32))
        self.conv4 = conv_block(32, 64, pool=True)
        self.res3 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        self.conv5 = conv_block(64, 128, pool=True)
        self.res4 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.classifier = nn.Sequential(nn.MaxPool2d(2),  # 512 x 1 x 1
                                        nn.Flatten(),  # 512
                                        nn.Linear(128, num_classes))  # 2056 -> 100

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.res2(out) + out
        out = self.conv4(out)
        out = self.res3(out) + out
        out = self.conv5(out)
        out = self.res4(out) + out
        out = self.classifier(out)
        return out
