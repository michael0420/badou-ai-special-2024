import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,padding):
        super(Conv2d_BN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.conv2d_bn = Conv2d_BN(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=0)
        self.conv2d_bn2 = Conv2d_BN(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=0)
        self.conv2d_bn3 = Conv2d_BN(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=(3-1)//2)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2)

        self.conv2d_bn4 = Conv2d_BN(in_channels=64,out_channels=80,kernel_size=1,stride=1,padding=0)
        self.conv2d_bn5 =Conv2d_BN(in_channels=80,out_channels=192,kernel_size=3,stride=1,padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2)  #192,35,35

        #block1 part1
        #1*1分支
        self.branch1x1_b1_p1 = Conv2d_BN(in_channels=192,out_channels=64,kernel_size=1,stride=1,padding=0)
        #1*1，5*5分支
        self.branch5x5_1x1_b1_p1 = Conv2d_BN(in_channels=192, out_channels=48, kernel_size=1, stride=1, padding=0)
        self.branch5x5_b1_p1 = Conv2d_BN(in_channels=48, out_channels=64, kernel_size=5,stride=1,padding=2)
        #1*1，3*3，3*3分支
        self.branch3X3_1x1_b1_p1 = Conv2d_BN(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch3x3_3x3_b1_p1 = Conv2d_BN(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.branch3x3_b1_p1 = Conv2d_BN(in_channels=96,out_channels=96,kernel_size=3,stride=1,padding=1)
        #3*3pool分支，1*1
        self.avgpool_b1_p1 =nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.avgpool_conv_b1_p1 = nn.Conv2d(in_channels=192,out_channels=32,kernel_size=1,stride=1,padding=0)
        #part2(此时的输入是35*35*256)
        self.branch1x1_b1_p2 = Conv2d_BN(in_channels=256,out_channels=64,kernel_size=1,stride=1,padding=0)

        self.branch5x5_1x1_b1_p2=Conv2d_BN(in_channels=256,out_channels=48,kernel_size=1,stride=1,padding=0)
        self.branch5x5_b1_p2 = Conv2d_BN(in_channels=48,out_channels=64,kernel_size=5,stride=1,padding=2)

        self.branch_3x3_1x1_b1_p2 =Conv2d_BN(in_channels=256,out_channels=64,kernel_size=1,stride=1,padding=0)
        self.branch_3x3_3x3_b1_p2 = Conv2d_BN(in_channels=64,out_channels=96,kernel_size=3,stride=1,padding=1)
        self.branch_3x3_b1_p2 = Conv2d_BN(in_channels=96,out_channels=96,kernel_size=3,stride=1,padding=1)

        self.avgpool_b1_p2 = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.avgpool_conv_b1_p2 = Conv2d_BN(in_channels=256,out_channels=64,kernel_size=1,stride=1,padding=0)
        #part3(此时的输入是35*35*288)
        #1x1
        self.branch_1x1_b1_p3 = Conv2d_BN(in_channels=288,out_channels=64,kernel_size=1,stride=1,padding=0)
        #1*1,5*5
        self.branch_5x5_1x1_b1_p3 = Conv2d_BN(in_channels=288,out_channels=48,kernel_size=1,stride=1,padding=0)
        self.branch_5x5_b1_p3 = Conv2d_BN(in_channels=48,out_channels=64,kernel_size=5,stride=1,padding=2)
        #1*1,3*3,3*3
        self.branch_3x3_1x1_b1_p3 = Conv2d_BN(in_channels=288,out_channels=64,kernel_size=1,stride=1,padding=0)
        self.branch_3x3_3x3_b1_p3 = Conv2d_BN(in_channels=64,out_channels=96,kernel_size=3,stride=1,padding=1)
        self.branch_3x3_b1_p3 = Conv2d_BN(in_channels=96,out_channels=96,kernel_size=3,stride=1,padding=1)

        #3*3poolsize,1*1卷积
        self.avgpool_b1_p3 = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.avgpool_conv_b1_p3 = nn.Conv2d(in_channels=288,out_channels=64,kernel_size=1,stride=1,padding=0)

        #block2 part1 (此时的输入是35*35*288)
        self.branch3x3_1_b2_p1 = Conv2d_BN(in_channels=288,out_channels=384,kernel_size=3,stride=2,padding=0)

        self.branch3x3_1X1_b2_p1 = Conv2d_BN(in_channels=288,out_channels=64,kernel_size=1,stride=1,padding=0)
        self.branch3x3_3x3_b2_p1 = Conv2d_BN(in_channels=64,out_channels=96,kernel_size=3,stride=1,padding=1)
        self.branch3x3_b2_p1 = Conv2d_BN(in_channels=96,out_channels=96,kernel_size=3,stride=2,padding=0)
        self.avgpool_b2_p1 = nn.AvgPool2d(kernel_size=3,stride=2,padding=0)
        #part2 (17*17*768)
        self.branch1x1_b2_p2 = Conv2d_BN(in_channels=768,out_channels=192,kernel_size=1,stride=1,padding=0)

        self.branch1x7_1x1_b2_p2 = Conv2d_BN(in_channels=768,out_channels=128,kernel_size=1,stride=1,padding=0)
        self.branch1x7_b2_p2 = Conv2d_BN(in_channels=128,out_channels=128,kernel_size=(1,7),stride=1,padding=(0,3))
        self.branch7x1_b2_p2 = Conv2d_BN(in_channels=128,out_channels=192,kernel_size=(7,1),stride=1,padding=(3,0))

        self.branch7x7_1_b2_p2 = Conv2d_BN(in_channels=768,out_channels=128,kernel_size=1,stride=1,padding=0)
        self.branch7x7_2_b2_p2 = Conv2d_BN(in_channels=128,out_channels=128,kernel_size=(7,1),stride=1,padding=(3,0))
        self.branch7x7_3_b2_p2 = Conv2d_BN(in_channels=128, out_channels=128, kernel_size=(1, 7), stride=1, padding=(0,3))
        self.branch7x7_4_b2_p2 = Conv2d_BN(in_channels=128,out_channels=128,kernel_size=(7,1),stride=1,padding=(3,0))
        self.branch7x7_5_b2_p2 = Conv2d_BN(in_channels=128, out_channels=192, kernel_size=(1, 7), stride=1, padding=(0,3))

        self.avgpool_b2_p2 = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.avgpool_conv_b2_p2 = nn.Conv2d(in_channels=768,out_channels=192,kernel_size=1,stride=1,padding=0)

        #part3 part4共用(17*17*768)
        self.branch1x1_b2_p3 = Conv2d_BN(in_channels=768,out_channels=192,kernel_size=1,stride=1,padding=0)

        self.branch1x7_1x1_b2_p3 = Conv2d_BN(in_channels=768,out_channels=160,kernel_size=1,stride=1,padding=0)
        self.branch1x7_b2_p3 = Conv2d_BN(in_channels=160,out_channels=160,kernel_size=(1,7),stride=1,padding=(0,3))
        self.branch7x1_b2_p3 = Conv2d_BN(in_channels=160,out_channels=192,kernel_size=(7,1),stride=1,padding=(3,0))

        self.branch7x7_1_b2_p3 = Conv2d_BN(in_channels=768,out_channels=160,kernel_size=(1,1),stride=1,padding=0)
        self.branch7x7_2_b2_p3 = Conv2d_BN(in_channels=160,out_channels=160,kernel_size=(7,1),stride=1,padding=(3,0))
        self.branch7x7_3_b2_p3 = Conv2d_BN(in_channels=160,out_channels=160,kernel_size=(1,7),stride=1,padding=(0,3))
        self.branch7x7_4_b2_p3 = Conv2d_BN(in_channels=160,out_channels=160,kernel_size=(7,1),stride=1,padding=(3,0))
        self.branch7x7_5_b2_p3 = Conv2d_BN(in_channels=160,out_channels=192,kernel_size=(1,7),stride=1,padding=(0,3))

        self.avgpool_b2_p3 = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.avgpool_conv_b2_p3 = nn.Conv2d(in_channels=768,out_channels=192,kernel_size=1,stride=1,padding=0)

        #part5 (17*17*768)
        self.branch1x1_b2_p5 = Conv2d_BN(in_channels=768, out_channels=192, kernel_size=1, stride=1, padding=0)

        self.branch1x7_1x1_b2_p5 = Conv2d_BN(in_channels=768, out_channels=192, kernel_size=1, stride=1, padding=0)
        self.branch1x7_b2_p5 = Conv2d_BN(in_channels=192, out_channels=192, kernel_size=(1, 7), stride=1,
                                         padding=(0, 3))
        self.branch7x1_b2_p5 = Conv2d_BN(in_channels=192, out_channels=192, kernel_size=(7, 1), stride=1,
                                         padding=(3, 0))

        self.branch7x7_1_b2_p5 = Conv2d_BN(in_channels=768, out_channels=192, kernel_size=(1, 1), stride=1, padding=0)
        self.branch7x7_2_b2_p5 = Conv2d_BN(in_channels=192, out_channels=192, kernel_size=(7, 1), stride=1,
                                           padding=(3, 0))
        self.branch7x7_3_b2_p5 = Conv2d_BN(in_channels=192, out_channels=192, kernel_size=(1, 7), stride=1,
                                           padding=(0, 3))
        self.branch7x7_4_b2_p5 = Conv2d_BN(in_channels=192, out_channels=192, kernel_size=(7, 1), stride=1,
                                           padding=(3, 0))
        self.branch7x7_5_b2_p5 = Conv2d_BN(in_channels=192, out_channels=192, kernel_size=(1, 7), stride=1,
                                           padding=(0, 3))

        self.avgpool_b2_p5 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool_conv_b2_p5 = nn.Conv2d(in_channels=768, out_channels=192, kernel_size=1, stride=1, padding=0)


        #block3  part1 (17,17,768)
        self.branch_3x3_1x1_b3_p1 = Conv2d_BN(in_channels=768,out_channels=192,kernel_size=1,stride=1,padding=0)
        self.branch_3x3_b3_p1 = Conv2d_BN(in_channels=192,out_channels=320,kernel_size=3,stride=2,padding=0)

        self.branch_1x7_1x1_b3_p1 = Conv2d_BN(in_channels=768,out_channels=192,kernel_size=1,stride=1,padding=0)
        self.branch_1x7_b3_p1 =Conv2d_BN(in_channels=192,out_channels=192,kernel_size=(1,7),stride=1,padding=(0,3))
        self.branch_7x1_b3_p1 =Conv2d_BN(in_channels=192,out_channels=192,kernel_size=(7,1),stride=1,padding=(3,0))
        self.branch_7x1_3x3_b3_p1 = Conv2d_BN(in_channels=192,out_channels=192,kernel_size=3,stride=2,padding=0)

        self.maxpool_b3_p1 = nn.MaxPool2d(kernel_size=3,stride=2)


        #part2  （8*8*1280）
        self.branch_1x1_b3_p2 = Conv2d_BN(in_channels=1280,out_channels=320,kernel_size=1,stride=1,padding=0)

        self.branch_1x3_1x1_b3_p2 = Conv2d_BN(in_channels=1280,out_channels=384,kernel_size=1,stride=1,padding=0)
        self.branch_1x3_b3_p2 = Conv2d_BN(in_channels=384,out_channels=384,kernel_size=(1,3),stride=1,padding=(0,1))
        self.branch_3x1_b3_p2 = Conv2d_BN(in_channels=384,out_channels=384,kernel_size=(3,1),stride=1,padding=(1,0))

        self.branch_3x3_1_b3_p2 = Conv2d_BN(in_channels=1280,out_channels=448,kernel_size=(1,1),stride=1,padding=0)
        self.branch_3x3_2_b3_p2 = Conv2d_BN(in_channels=448,out_channels=384,kernel_size=(3,3),stride=1,padding=1)
        self.branch_3x3_3_b3_p2 = Conv2d_BN(in_channels=384,out_channels=384,kernel_size=(1,3),stride=1,padding=(0,1))
        self.branch_3x3_4_b3_p2 = Conv2d_BN(in_channels=384, out_channels=384, kernel_size=(3,1), stride=1,padding=(1,0))

        self.avgpool_b3_p2 = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.avgpool_conv_b3_p2 = nn.Conv2d(in_channels=1280,out_channels=192,kernel_size=1,stride=1,padding=0)

        #part3  (8*8*2048)
        self.branch_1x1_b3_p3 = Conv2d_BN(in_channels=2048,out_channels=320,kernel_size=1,stride=1,padding=0)

        self.branch_1x3_1x1_b3_p3 = Conv2d_BN(in_channels=2048,out_channels=384,kernel_size=1,stride=1,padding=0)
        self.branch_1x3_b3_p3 = Conv2d_BN(in_channels=384,out_channels=384,kernel_size=(1,3),stride=1,padding=(0,1))
        self.branch_3x1_b3_p3 = Conv2d_BN(in_channels=384,out_channels=384,kernel_size=(3,1),stride=1,padding=(1,0))

        self.branch_3x3_1_b3_p3 = Conv2d_BN(in_channels=2048,out_channels=448,kernel_size=(1,1),stride=1,padding=0)
        self.branch_3x3_2_b3_p3 = Conv2d_BN(in_channels=448,out_channels=384,kernel_size=(3,3),stride=1,padding=1)
        self.branch_3x3_3_b3_p3 = Conv2d_BN(in_channels=384,out_channels=384,kernel_size=(1,3),stride=1,padding=(0,1))
        self.branch_3x3_4_b3_p3 = Conv2d_BN(in_channels=384, out_channels=384, kernel_size=(3,1), stride=1,padding=(1,0))

        self.avgpool_b3_p3 = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.avgpool_conv_b3_p3 = nn.Conv2d(in_channels=2048,out_channels=192,kernel_size=1,stride=1,padding=0)


        #
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(2048,10)





    def forward(self, x):
        x = self.conv2d_bn(x)    #149,149,32
        x = self.conv2d_bn2(x)  #147,147,32
        x = self.conv2d_bn3(x)   #147,147,64
        x = self.maxpool(x)     #73,73,64
        x = self.conv2d_bn4(x)  #73,73, 80
        x = self.conv2d_bn5(x)  #71,71,192
        x = self.maxpool2(x)   #35,35,192

        #block1
        #分支1*1
        x_1x1_b1_p1 = self.branch1x1_b1_p1(x)  #35,35,64
        #分支1*1，5*5
        x_5x5_1x1_b1_p1 = self.branch5x5_1x1_b1_p1(x)  #35,35,48
        x_5x5_b1_p1 = self.branch5x5_b1_p1(x_5x5_1x1_b1_p1) #35,35,64
        #分支1*1，3*3，3*3
        x_3x3_1x1_b1_p1 = self.branch3X3_1x1_b1_p1(x)    #35,35,64
        x_3x3_3x3_b1_p1 = self.branch3x3_3x3_b1_p1(x_3x3_1x1_b1_p1)  # 35,35,96
        x_3x3_b1_p1 = self.branch3x3_b1_p1(x_3x3_3x3_b1_p1)   #35,35,96
        #分支pool 3*3，1*1
        x_avgpool_b1_p1 = self.avgpool_b1_p1(x)
        x_avgpool_conv_b1_p1 = self.avgpool_conv_b1_p1(x_avgpool_b1_p1) #35,35,32
        #合并，64+64+96+32=256
        x = torch.cat([x_1x1_b1_p1,x_5x5_b1_p1,x_3x3_b1_p1,x_avgpool_conv_b1_p1],dim=1) #35,35,256

        #part2
        x_1x1_b1_p2 = self.branch1x1_b1_p2(x)    #35,35,64

        x_5x5_1x1_b1_p2 = self.branch5x5_1x1_b1_p2(x)  #35,35,48
        x_5x5_b1_p2 = self.branch5x5_b1_p2(x_5x5_1x1_b1_p2) #35,35,64

        x_3x3_1x1_b1_p2 = self.branch_3x3_1x1_b1_p2(x)   #35,35,64
        x_3x3_3x3_b1_p2 = self.branch_3x3_3x3_b1_p2(x_3x3_1x1_b1_p2) #35,35,96
        x_3x3_b1_p2 = self.branch_3x3_b1_p2(x_3x3_3x3_b1_p2)  #35,35,96

        x_avgpool_b1_p2 = self.avgpool_b1_p2(x)
        x_avgpool_conv_b1_p2 = self.avgpool_conv_b1_p2(x_avgpool_b1_p2)  #35,35,64

        x = torch.cat([x_1x1_b1_p2,x_5x5_b1_p2,x_3x3_b1_p2,x_avgpool_conv_b1_p2],dim=1) #35,35,288

        #part3
        x_1x1_b1_p3 = self.branch_1x1_b1_p3(x)  #35,35,64

        x_5x5_1x1_b1_p3 = self.branch_5x5_1x1_b1_p3(x)    #35,35,48
        x_5x5_b1_p3 =self.branch_5x5_b1_p3(x_5x5_1x1_b1_p3) #35,35,64

        x_3x3_1x1_b1_p3 = self.branch_3x3_1x1_b1_p3(x)
        x_3x3_3x3_b1_p3 = self.branch_3x3_3x3_b1_p3(x_3x3_1x1_b1_p3)
        x_3x3_b1_p3 = self.branch_3x3_b1_p3(x_3x3_3x3_b1_p3)    #35,35,96

        x_avgpool_b1_p3 = self.avgpool_b1_p3(x)
        x_avgpool_conv_b1_p3 = self.avgpool_conv_b1_p3(x_avgpool_b1_p3)  #35,35,64
        #64+64+96+64 = 288
        x = torch.cat([x_1x1_b1_p3,x_5x5_b1_p3,x_3x3_b1_p3,x_avgpool_conv_b1_p3],dim=1)


        #block2 part1
        x_3x3_1_b2_p1 = self.branch3x3_1_b2_p1(x)  #17,17,384

        x_3x3_1x1_b2_p1 = self.branch3x3_1X1_b2_p1(x)
        x_3x3_3x3_b2_p1 = self.branch3x3_3x3_b2_p1(x_3x3_1x1_b2_p1)
        x_3x3_b2_p1 = self.branch3x3_b2_p1(x_3x3_3x3_b2_p1)    #17,17,96

        x_avgpool_b2_p1 = self.avgpool_b2_p1(x)  #17,17,288
        x = torch.cat([x_3x3_1_b2_p1,x_3x3_b2_p1,x_avgpool_b2_p1],dim=1)
        #part2
        x_1x1_b2_p2 = self.branch1x1_b2_p2(x)   #17.17.192

        x_1x7_1x1_b2_p2 = self.branch1x7_1x1_b2_p2(x)
        x_1x7_b2_p2 = self.branch1x7_b2_p2(x_1x7_1x1_b2_p2)
        x_7x1_b2_p2 = self.branch7x1_b2_p2(x_1x7_b2_p2)   #17.17.192

        x_7x7_1_b2_p2 = self.branch7x7_1_b2_p2(x)
        x_7x7_2_b2_p2 = self.branch7x7_2_b2_p2(x_7x7_1_b2_p2)
        x_7x7_3_b2_p2 = self.branch7x7_3_b2_p2(x_7x7_2_b2_p2)
        x_7x7_4_b2_p2 = self.branch7x7_4_b2_p2(x_7x7_3_b2_p2)
        x_7x7_5_b2_p2 = self.branch7x7_5_b2_p2(x_7x7_4_b2_p2)  #17,17,192

        x_avgpool_b2_p2 =self.avgpool_b2_p2(x)
        x_avgpool_conv_b2_p2 = self.avgpool_conv_b2_p2(x_avgpool_b2_p2)  #17,17,192

        x = torch.cat([x_1x1_b2_p2,x_7x1_b2_p2,x_7x7_5_b2_p2,x_avgpool_conv_b2_p2],dim=1) #17,17,768

        #part3
        x_branch1x1_b2_p3 = self.branch1x1_b2_p3(x)     #17,17,192

        x_branch1x7_1x1_b2_p3 =self.branch1x7_1x1_b2_p3(x)
        x_branch1x7_b2_p3 = self.branch1x7_b2_p3(x_branch1x7_1x1_b2_p3)
        x_branch7x1_b2_p3 = self.branch7x1_b2_p3(x_branch1x7_b2_p3)   #17,17,192

        x_branch7x7_1_b2_p3 = self.branch7x7_1_b2_p3(x)
        x_branch7x7_2_b2_p3 = self.branch7x7_2_b2_p3(x_branch7x7_1_b2_p3)
        x_branch7x7_3_b2_p3 = self.branch7x7_3_b2_p3(x_branch7x7_2_b2_p3)
        x_branch7x7_4_b2_p3 = self.branch7x7_4_b2_p3(x_branch7x7_3_b2_p3)
        x_branch7x7_5_b2_p3 = self.branch7x7_5_b2_p3(x_branch7x7_4_b2_p3)  #17,17,192

        x_avgpool_b2_p3 = self.avgpool_b2_p3(x)
        x_avgpool_conv_b2_p3 = self.avgpool_conv_b2_p3(x_avgpool_b2_p3)   #17,17,192

        x = torch.cat([x_branch1x1_b2_p3,x_branch7x1_b2_p3,x_branch7x7_5_b2_p3,x_avgpool_conv_b2_p3],dim=1)

        # part4
        x_branch1x1_b2_p4 = self.branch1x1_b2_p3(x)     #17,17,192

        x_branch1x7_1x1_b2_p4 =self.branch1x7_1x1_b2_p3(x)
        x_branch1x7_b2_p4 = self.branch1x7_b2_p3(x_branch1x7_1x1_b2_p4)
        x_branch7x1_b2_p4 = self.branch7x1_b2_p3(x_branch1x7_b2_p4)   #17,17,192

        x_branch7x7_1_b2_p4 = self.branch7x7_1_b2_p3(x)
        x_branch7x7_2_b2_p4 = self.branch7x7_2_b2_p3(x_branch7x7_1_b2_p4)
        x_branch7x7_3_b2_p4 = self.branch7x7_3_b2_p3(x_branch7x7_2_b2_p4)
        x_branch7x7_4_b2_p4 = self.branch7x7_4_b2_p3(x_branch7x7_3_b2_p4)
        x_branch7x7_5_b2_p4 = self.branch7x7_5_b2_p3(x_branch7x7_4_b2_p4)  #17,17,192

        x_avgpool_b2_p4 = self.avgpool_b2_p3(x)
        x_avgpool_conv_b2_p4 = self.avgpool_conv_b2_p3(x_avgpool_b2_p4)   #17,17,192

        x = torch.cat([x_branch1x1_b2_p4,x_branch7x1_b2_p4,x_branch7x7_5_b2_p4,x_avgpool_conv_b2_p4],dim=1)

        #part5
        x_branch1x1_b2_p5 = self.branch1x1_b2_p5(x)  # 17,17,192

        x_branch1x7_1x1_b2_p5 = self.branch1x7_1x1_b2_p5(x)
        x_branch1x7_b2_p5 = self.branch1x7_b2_p5(x_branch1x7_1x1_b2_p5)
        x_branch7x1_b2_p5 = self.branch7x1_b2_p5(x_branch1x7_b2_p5)  # 17,17,192

        x_branch7x7_1_b2_p5 = self.branch7x7_1_b2_p5(x)
        x_branch7x7_2_b2_p5 = self.branch7x7_2_b2_p5(x_branch7x7_1_b2_p5)
        x_branch7x7_3_b2_p5 = self.branch7x7_3_b2_p5(x_branch7x7_2_b2_p5)
        x_branch7x7_4_b2_p5 = self.branch7x7_4_b2_p5(x_branch7x7_3_b2_p5)
        x_branch7x7_5_b2_p5 = self.branch7x7_5_b2_p5(x_branch7x7_4_b2_p5)  # 17,17,192

        x_avgpool_b2_p5 = self.avgpool_b2_p5(x)
        x_avgpool_conv_b2_p5 = self.avgpool_conv_b2_p5(x_avgpool_b2_p5)  # 17,17,192

        x = torch.cat([x_branch1x1_b2_p5, x_branch7x1_b2_p5, x_branch7x7_5_b2_p5, x_avgpool_conv_b2_p5], dim=1)



        #block3 part1
        x_3x3_1x1_b3_p1 = self.branch_3x3_1x1_b3_p1(x)
        x_3x3_b3_p1 = self.branch_3x3_b3_p1(x_3x3_1x1_b3_p1)  #3*3*320

        x_1x7_1x1_b3_p1 = self.branch_1x7_1x1_b3_p1(x)
        x_1x7_b3_p1 = self.branch_1x7_b3_p1(x_1x7_1x1_b3_p1)
        x_7x1_b3_p1 = self.branch_7x1_b3_p1(x_1x7_b3_p1)
        x_7x1_3x3_b3_p1 = self.branch_7x1_3x3_b3_p1(x_7x1_b3_p1)  #8*8*192

        x_maxpool_b3_p1 = self.maxpool_b3_p1(x)    #8*8*768

        x = torch.cat([x_3x3_b3_p1,x_7x1_3x3_b3_p1,x_maxpool_b3_p1],dim=1) # 8*8*1280


        #part2
        x_1x1_b3_p2 = self.branch_1x1_b3_p2(x)   #8*8*320

        x_1x3_1x1_b3_p2 = self.branch_1x3_1x1_b3_p2(x)
        x_1x3_b3_p2 = self.branch_1x3_b3_p2(x_1x3_1x1_b3_p2)
        x_3x1_b3_p2 = self.branch_3x1_b3_p2(x_1x3_1x1_b3_p2)
        x_branch3x3_1 = torch.cat([x_1x3_b3_p2,x_3x1_b3_p2],dim=1)    #8*8*768

        x_3x3_1_b3_p2 = self.branch_3x3_1_b3_p2(x)
        x_3x3_2_b3_p2 = self.branch_3x3_2_b3_p2(x_3x3_1_b3_p2)
        x_3x3_3_b3_p2 = self.branch_3x3_3_b3_p2(x_3x3_2_b3_p2)
        x_3x3_4_b3_p2 = self.branch_3x3_4_b3_p2(x_3x3_2_b3_p2)
        x_branch3x3_2 = torch.cat([x_3x3_3_b3_p2,x_3x3_4_b3_p2],dim=1)  #8*8*768

        x_avgpool_b3_p2 = self.avgpool_b3_p2(x)
        x_avgpool_conv_b3_p2 = self.avgpool_conv_b3_p2(x_avgpool_b3_p2)   #8*8*192

        #8*8*2048
        x = torch.cat([x_1x1_b3_p2,x_branch3x3_1,x_branch3x3_2,x_avgpool_conv_b3_p2],dim=1)


        # part3 输入是8*8*2048
        x_1x1_b3_p3 = self.branch_1x1_b3_p3(x)  # 8*8*320

        x_1x3_1x1_b3_p3 = self.branch_1x3_1x1_b3_p3(x)
        x_1x3_b3_p3 = self.branch_1x3_b3_p3(x_1x3_1x1_b3_p3)
        x_3x1_b3_p3 = self.branch_3x1_b3_p3(x_1x3_1x1_b3_p3)
        x_branch3x3_p3 = torch.cat([x_1x3_b3_p3, x_3x1_b3_p3], dim=1)  # 8*8*768

        x_3x3_1_b3_p3 = self.branch_3x3_1_b3_p3(x)
        x_3x3_2_b3_p3 = self.branch_3x3_2_b3_p3(x_3x3_1_b3_p3)
        x_3x3_3_b3_p3 = self.branch_3x3_3_b3_p3(x_3x3_2_b3_p3)
        x_3x3_4_b3_p3 = self.branch_3x3_4_b3_p3(x_3x3_2_b3_p3)
        x_branch3x3_p3_1 = torch.cat([x_3x3_3_b3_p3, x_3x3_4_b3_p3], dim=1)  # 8*8*768

        x_avgpool_b3_p3 = self.avgpool_b3_p3(x)
        x_avgpool_conv_b3_p3 = self.avgpool_conv_b3_p3(x_avgpool_b3_p3)  # 8*8*192

        # 8*8*2048
        x = torch.cat([x_1x1_b3_p3, x_branch3x3_p3, x_branch3x3_p3_1, x_avgpool_conv_b3_p3], dim=1)

        x =self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        x= self.linear(x)

        return x




if __name__ == '__main__':
    x = torch.randn((1,3,299,299))
    model = InceptionV3()
    y = model(x)
    print(y.shape)
