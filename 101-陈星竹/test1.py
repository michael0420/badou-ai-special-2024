import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

sample_nums = 100
mean_value = 1.7
bias = 1
torch.manual_seed(10)
n_data = torch.ones(sample_nums, 2)
x0 = torch.normal(mean_value * n_data, 1) + bias     #类别0 数据 shape=(100, 2)
y0 = torch.zeros(sample_nums,1)                     # 类别0 标签 shape=(100, 1)
x1 = torch.normal(-mean_value * n_data, 1) + bias    #类别1 数据 shape=(100, 2)
y1 = torch.ones(sample_nums,1)                      #类别1 标签 shape=(100, 1)
train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0)

#绘制数据
plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')
plt.legend()
plt.show()


# 利用torch.nn实现逻辑回归
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x


logistic_model = LogisticRegression()
# if torch.cuda.is_available():
#     logistic_model.cuda()

# loss函数和优化
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(logistic_model.parameters(), lr=0.01, momentum=0.9)
# 开始训练
# 训练10000次
for epoch in range(10000):
    #     if torch.cuda.is_available():
    #         x_data=Variable(x).cuda()
    #         y_data=Variable(y).cuda()
    #     else:
    #         x_data=Variable(x)
    #         y_data=Variable(y)

    out = logistic_model(train_x)  # 根据逻辑回归模型拟合出的y值
    loss = criterion(out.squeeze(), train_y)  # 计算损失函数
    print_loss = loss.data.item()  # 得出损失函数值
    # 反向传播
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    mask = out.ge(0.5).float()  # 以0.5为阈值进行分类
    correct = (mask == train_y).sum().squeeze()  # 计算正确预测的样本个数
    acc = correct.item() / train_x.size(0)  # 计算精度
    # 每隔20轮打印一下当前的误差和精度
    if (epoch + 1) % 100 == 0:
        print('*' * 10)
        print('epoch {}'.format(epoch + 1))  # 误差
        print('loss is {:.4f}'.format(print_loss))
        print('acc is {:.4f}'.format(acc))  # 精度

w0, w1 = logistic_model.lr.weight[0]
w0 = float(w0.item())
w1 = float(w1.item())
b = float(logistic_model.lr.bias.item())
plot_x = np.arange(-7, 7, 0.1)
plot_y = (-w0 * plot_x - b) / w1
plt.xlim(-5, 7)
plt.ylim(-7, 7)
plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')
plt.plot(plot_x, plot_y)
plt.show()
