# 方法1
from torch import nn

model = nn.Sequential()
model.add_module('fc1', nn.Linear(3,4)) # 添加全连接层, 输入3个神经元，输出4个神经元
model.add_module('fc2', nn.Linear(4,2)) # 添加全连接层, 输入4个神经元，输出2个神经元
model.add_module('output', nn.Softmax(2)) # 添加输出层，即Softmax层，输出2个神经元

# 方法2
model2 = nn.Sequential( # 一个包含2个卷积层的神经网络
          nn.Conv2d(1,20,5), # 输入通道数为1，输出通道数为20，卷积核大小为5
          nn.ReLU(), # 使用ReLU激活函数
          nn.Conv2d(20,64,5), # 输入通道数为20，输出通道数为64，卷积核大小为5
          nn.ReLU()
        )
# 方法3        
model3 = nn.ModuleList([nn.Linear(3,4), nn.ReLU(), nn.Linear(4,2)]) # 一个包含2个全连接层的神经网络,个包含2个全连接层的神经网络, 输入3个神经元，输出4个神经元，再输出2个神经元
