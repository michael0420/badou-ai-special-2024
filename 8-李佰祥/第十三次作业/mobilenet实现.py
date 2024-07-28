import torch
import torch.nn as nn


class Conv_block(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Depthwise_conv_block(nn.Module):
    def __init__(self, input_channels, pointwise_conv_filters,depth_multiplier,kernel_size, stride, padding):
        super(Depthwise_conv_block, self).__init__()
        #当 groups 大于 1 时，输入通道被分成 groups 组，每组有一个独立的卷积核
        #深度卷积的运作方式：在深度卷积中，每个输入通道都有一个独立的卷积核。这意味着如果输入有 C 个通道，那么会有 C 个卷积核，每个卷积核只处理一个输入通道。因此，每个输入通道会产生一个输出通道。
        #depth_multiplier 的作用：depth_multiplier 是一个超参数，
        # 用于控制每个输入通道被多少个独立的卷积核处理。当 depth_multiplier 设置为 1 时，
        # 每个输入通道会被一个卷积核处理，输出通道数等于输入通道数。
        # 但是，如果 depth_multiplier 大于 1，那么每个输入通道会被 depth_multiplier 个卷积核处理，
        # 因此输出通道数会是 in_channels * depth_multiplier。
        #这就是深度卷积的核心概念——每个输入通道独立地被卷积，而不与其他通道混合
        #深度卷积
        self.depthwise = nn.Conv2d(in_channels=input_channels,out_channels=input_channels*depth_multiplier,groups=input_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(input_channels*depth_multiplier)
        self.relu = nn.ReLU(inplace=True)

        #点卷积（pointwise）
        self.pointwise =nn.Conv2d(input_channels*depth_multiplier,pointwise_conv_filters,kernel_size=1,stride=1,padding=0)
        self.bn2 = nn.BatchNorm2d(pointwise_conv_filters)
        self.relu2 = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x



class MobileNet(nn.Module):
    def __init__(self, depth_multiplier=1.0, input_channels=3, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv = Conv_block(input_channels, 32, kernel_size=3, stride=2, padding=1)
        #输出112，112，64
        self.depthwise112x112x64 =Depthwise_conv_block(32, pointwise_conv_filters=64,depth_multiplier=1, kernel_size=3, stride=1, padding=1)

        self.depthwise56x56x128 = Depthwise_conv_block(64,128,1,stride=2,kernel_size=3,padding=1)
        self.depthwise56x56x128_2 = Depthwise_conv_block(128, pointwise_conv_filters=128, depth_multiplier=1,
                                                        kernel_size=3, stride=1, padding=1)

        self.depthwise28x28x256 = Depthwise_conv_block(128, 256, 1,stride=2,kernel_size=3,padding=1)
        self.depthwise28x28x256_2 = Depthwise_conv_block(256, 256, 1,stride=1,kernel_size=3,padding=1)

        self.depthwise14x14x512 = Depthwise_conv_block(256, 512, 1,stride=2,kernel_size=3,padding=1)

        self.depthwise14x14x512_1= Depthwise_conv_block(512, 512, 1,stride=1,kernel_size=3,padding=1)
        self.depthwise14x14x512_2 = Depthwise_conv_block(512, 512, 1, stride=1, kernel_size=3, padding=1)
        self.depthwise14x14x512_3 = Depthwise_conv_block(512, 512, 1, stride=1, kernel_size=3, padding=1)
        self.depthwise14x14x512_4 = Depthwise_conv_block(512, 512, 1, stride=1, kernel_size=3, padding=1)
        self.depthwise14x14x512_5 = Depthwise_conv_block(512, 512, 1, stride=1, kernel_size=3, padding=1)

        self.depthwise7x7x1024 = Depthwise_conv_block(512, 1024, 1, stride=2, kernel_size=3,padding=1)
        self.depthwise7x7x1024_1 = Depthwise_conv_block(1024, 1024, 1, stride=1, kernel_size=3, padding=1)


        self.adaptiveavgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1e-3)
        self.conv2 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)





    def forward(self, x):
        x = self.conv(x)
        x = self.depthwise112x112x64(x)
        x = self.depthwise56x56x128(x)
        x = self.depthwise56x56x128_2(x)
        x = self.depthwise28x28x256(x)
        x = self.depthwise28x28x256_2(x)
        x = self.depthwise14x14x512(x)
        x = self.depthwise14x14x512_1(x)
        x = self.depthwise14x14x512_2(x)
        x = self.depthwise14x14x512_3(x)
        x = self.depthwise14x14x512_4(x)
        x = self.depthwise14x14x512_5(x)
        x = self.depthwise7x7x1024(x)
        x = self.depthwise7x7x1024_1(x)

        x = self.adaptiveavgpool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.softmax(x)
        x = x.view(x.size(0), -1)
        return x



if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = MobileNet()
    res = model(x)
    print(res.shape)
