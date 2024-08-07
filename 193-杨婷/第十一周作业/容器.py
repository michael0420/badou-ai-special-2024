# 运行不了，示例
# 方法1
model = nn.Sequential()  # 空序列模型
model.add_module('fc1', nn.Linear(3, 4))  # 行列要对应
model.add_module('fc2', nn.Linear(4, 2))
model.add_module('output', nn.Softmax(2))

# 方法2
model2 = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
# 方法3        
model3 = nn.ModuleList([nn.Linear(3,4), nn.ReLU(), nn.Linear(4,2)])
