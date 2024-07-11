import torch.nn as nn

# 第一种定义方法
model = nn.Sequential()
# 输入层三个节点，隐藏层四个节点
model.add_module('fc1', nn.Linear(10, 20))
# 隐藏层四个节点，输出层三个节点
model.add_module('fc2', nn.Linear(20, 3))
model.add_module('output', nn.Softmax(2))

# 第二种定义方法
model2 = nn.Sequential(nn.Conv2d(10, 10, 5), nn.ReLU(), nn.Conv2d(10, 5, 5), nn.ReLU())

# 第三种定义方法
model3 = nn.ModuleList([nn.Linear(10, 10, 5), nn.ReLU(), nn.Linear(10, 20, 5), nn.ReLU()])
