import torch
import torch.nn as nn
from functools import partial

# functools.partial()：减少某个函数的参数个数。 partial() 函数允许你给一个或多个参数设置固定的值，减少接下来被调用时的参数个数

'''-----------------------第一步：定义卷积模块-----------------------'''


# 基础卷积模块
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output=False):
        super(Conv2d, self).__init__()
        '''卷积层'''
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        '''输出层'''
        self.output = output
        if self.output == False:
            '''bn层'''
            self.bn = nn.BatchNorm2d(out_channels)

            '''relu层'''
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.output:
            return x
        else:
            x = self.bn(x)
            x = self.relu(x)
            return x


class Separable_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Separable_Conv2d, self).__init__()
        self.conv_h = nn.Conv2d(in_channels, in_channels, (kernel_size, 1), stride=(stride, 1), padding=(padding, 0))
        self.conv_w = nn.Conv2d(in_channels, out_channels, (1, kernel_size), stride=(1, stride), padding=(0, padding))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_h(x)
        x = self.conv_w(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Concat_Separable_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Concat_Separable_Conv2d, self).__init__()
        self.conv_h = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), stride=(stride, 1), padding=(padding, 0))
        self.conv_w = nn.Conv2d(in_channels, out_channels, (1, kernel_size), stride=(1, stride), padding=(0, padding))
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h = self.conv_h(x)
        x_w = self.conv_w(x)
        x = torch.cat([x_h, x_w], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Flatten（）就是将2D的特征图压扁为1D的特征向量，是展平操作，进入全连接层之前使用,类才能写进nn.Sequential
class Flatten(nn.Module):
    # 传入输入维度和输出维度
    def __init__(self):
        # 调用父类构造函数
        super(Flatten, self).__init__()

    # 实现forward函数
    def forward(self, x):
        # 保存batch维度，后面的维度全部压平
        return torch.flatten(x, 1)


# Squeeze（）降维
class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)


'''-----------------------搭建GoogLeNet网络--------------------------'''


class GoogLeNet(nn.Module):
    def __init__(self, num_classes, mode='train'):
        super(GoogLeNet, self).__init__()
        self.num_classes = num_classes
        self.mode = mode
        self.layers = nn.Sequential(
            Conv2d(3, 32, 3, stride=2),
            Conv2d(32, 32, 3, stride=1),
            Conv2d(32, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 80, kernel_size=3),
            Conv2d(80, 192, kernel_size=3, stride=2),
            Conv2d(192, 288, kernel_size=3, stride=1, padding=1),
            # 输入：35*35*288。将5*5用两个3*3代替
            Inceptionv3(288, 64, 48, 64, 64, 96, 64, mode='1'),  # 3a
            Inceptionv3(288, 64, 48, 64, 64, 96, 64, mode='1'),  # 3b
            Inceptionv3(288, 0, 128, 384, 64, 96, 0, stride=2, pool_type='MAX', mode='1'),  # 3c
            # 输入：17*17*768。
            Inceptionv3(768, 192, 128, 192, 128, 192, 192, mode='2'),  # 4a
            Inceptionv3(768, 192, 160, 192, 160, 192, 192, mode='2'),  # 4b
            Inceptionv3(768, 192, 160, 192, 160, 192, 192, mode='2'),  # 4c
            Inceptionv3(768, 192, 192, 192, 192, 192, 192, mode='2'),  # 4d
            Inceptionv3(768, 0, 192, 320, 192, 192, 0, stride=2, pool_type='MAX', mode='2'),  # 4e
            # 8*8*1280
            Inceptionv3(1280, 320, 384, 384, 448, 384, 192, mode='3'),  # 5a
            Inceptionv3(2048, 320, 384, 384, 448, 384, 192, pool_type='MAX', mode='3'),  # 5b
            nn.AvgPool2d(8, 1),
            Conv2d(2048, num_classes, kernel_size=1, output=True),
            Squeeze(),
        )
        if mode == 'train':
            self.aux = InceptionAux(768, num_classes)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            if (idx == 14 and self.mode == 'train'):
                aux = self.aux(x)
            x = layer(x)
        if self.mode == 'train':
            return x, aux
        else:
            return x

    '''-----------------------网络结构参数初始化--------------------------'''

    # 目的：使网络更好收敛，准确率更高
    def _initialize_weights(self):  # 将各种初始化方法定义为一个initialize_weights()的函数并在模型初始后进行使用。

        # 遍历网络中的每一层
        for m in self.modules():
            # isinstance(object, type)，如果指定的对象拥有指定的类型，则isinstance()函数返回True

            '''如果是卷积层Conv2d'''
            if isinstance(m, nn.Conv2d):
                # Kaiming正态分布方式的权重初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                '''判断是否有偏置：'''
                # 如果偏置不是0，将偏置置成0，对偏置进行初始化
                if m.bias is not None:
                    # torch.nn.init.constant_(tensor, val)，初始化整个矩阵为常数val
                    nn.init.constant_(m.bias, 0)

                '''如果是全连接层'''
            elif isinstance(m, nn.Linear):
                # init.normal_(tensor, mean=0.0, std=1.0)，使用从正态分布中提取的值填充输入张量
                # 参数：tensor：一个n维Tensor，mean：正态分布的平均值，std：正态分布的标准差
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


'''---------------------Inceptionv3-------------------------------------'''
'''
Inceptionv3由三个连续的Inception模块组组成
'''


class Inceptionv3(nn.Module):
    def __init__(self, input_channel, conv1_channel, conv3_reduce_channel,
                 conv3_channel, conv3_double_reduce_channel, conv3_double_channel, pool_reduce_channel, stride=1,
                 pool_type='AVG', mode='1'):

        super(Inceptionv3, self).__init__()
        self.stride = stride

        if stride == 2:
            padding_conv3 = 0
            padding_conv7 = 2
        else:
            padding_conv3 = 1
            padding_conv7 = 3
        if conv1_channel != 0:
            self.conv1 = Conv2d(input_channel, conv1_channel, kernel_size=1)
        else:
            self.conv1 = None
        self.conv3_reduce = Conv2d(input_channel, conv3_reduce_channel, kernel_size=1)
        # 第一种Inception模式：输入的特征图尺寸为35x35x288,采用了论文中图5中的架构，将5x5以两个3x3代替。
        if mode == '1':
            self.conv3 = Conv2d(conv3_reduce_channel, conv3_channel, kernel_size=3, stride=stride,
                                padding=padding_conv3)
            self.conv3_double1 = Conv2d(conv3_double_reduce_channel, conv3_double_channel, kernel_size=3, padding=1)
            self.conv3_double2 = Conv2d(conv3_double_channel, conv3_double_channel, kernel_size=3, stride=stride,
                                        padding=padding_conv3)

        # 第二种Inception模块：输入特征图尺寸为17x17x768，采用了论文中图6中nx1+1xn的不对称卷积结构
        elif mode == '2':
            self.conv3 = Separable_Conv2d(conv3_reduce_channel, conv3_channel, kernel_size=7, stride=stride,
                                          padding=padding_conv7)
            self.conv3_double1 = Separable_Conv2d(conv3_double_reduce_channel, conv3_double_channel, kernel_size=7,
                                                  padding=3)
            self.conv3_double2 = Separable_Conv2d(conv3_double_channel, conv3_double_channel, kernel_size=7,
                                                  stride=stride, padding=padding_conv7)

        # 第三种Inception模块：输入特征图尺寸为8x8x1280, 采用了论文图7中所示的并行模块的结构
        elif mode == '3':
            self.conv3 = Concat_Separable_Conv2d(conv3_reduce_channel, conv3_channel, kernel_size=3, stride=stride,
                                                 padding=1)
            self.conv3_double1 = Conv2d(conv3_double_reduce_channel, conv3_double_channel, kernel_size=3, padding=1)
            self.conv3_double2 = Concat_Separable_Conv2d(conv3_double_channel, conv3_double_channel, kernel_size=3,
                                                         stride=stride, padding=1)

        self.conv3_double_reduce = Conv2d(input_channel, conv3_double_reduce_channel, kernel_size=1)
        if pool_type == 'MAX':
            self.pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=padding_conv3)
        elif pool_type == 'AVG':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=padding_conv3)
        if pool_reduce_channel != 0:
            self.pool_reduce = Conv2d(input_channel, pool_reduce_channel, kernel_size=1)
        else:
            self.pool_reduce = None

    def forward(self, x):

        output_conv3 = self.conv3(self.conv3_reduce(x))
        output_conv3_double = self.conv3_double2(self.conv3_double1(self.conv3_double_reduce(x)))
        if self.pool_reduce != None:
            output_pool = self.pool_reduce(self.pool(x))
        else:
            output_pool = self.pool(x)

        if self.conv1 != None:
            output_conv1 = self.conv1(x)
            outputs = torch.cat([output_conv1, output_conv3, output_conv3_double, output_pool], dim=1)
        else:
            outputs = torch.cat([output_conv3, output_conv3_double, output_pool], dim=1)
        return outputs


'''------------辅助分类器---------------------------'''


class InceptionAux(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(InceptionAux, self).__init__()
        self.layers = nn.Sequential(
            nn.AvgPool2d(5, 3),
            Conv2d(input_channel, 128, 1),
            Conv2d(128, 1024, kernel_size=5),
            Conv2d(1024, num_classes, kernel_size=1, output=True),
            Squeeze()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


'''-------------------显示网络结构-------------------------------'''
if __name__ == '__main__':
    net = GoogLeNet(1000).cuda()
    from torchsummary import summary

    summary(net, (3, 299, 299))