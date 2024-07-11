#参考文档：https://blog.csdn.net/weixin_66845445/article/details/133828686
import numpy as np
import random as rd

#激活函数tanh及导数(tanh函数的导数在任一点上的值等于1减去该点上tanh函数值的平方 )
def sigmoid(x):
    return  np.tanh(x)

def  sigmoid_derivatives(x):
    return (1-(np.tanh(x)**2))

#生成【a,b】区间内的随机数
def random_num(a,b):
    return (b-a)*rd.random()+a   #random.random  生成0-1之间的浮点数

#生成一个m*n的零矩阵
def makematrix(m,n):
    matrix=np.zeros((m,n))
    return matrix

#构造三层BP神经网络架构
class BP:
#初始化函数：各层结点数、激活结点、权重矩阵、偏差、动量因子
    def __init__(self,num_in,num_hidden,num_out):
        self.num_in=num_in+1   #输入层结点数  并增加一个偏置结点（阈值）
        self.num_hidden=num_hidden+1  #隐藏层结点数  并增加一个偏置结点（阈值）
        self.num_out=num_out  #输出层结点数

        #激活BP神经网络的所有结点（向量）
        self.active_in=np.array([-1.0]*self.num_in)
        self.active_hidden=np.array([-1.0]*self.num_hidden)
        self.active_out=np.array([1.0]*self.num_out)

        #创建权重矩阵
        self.weight_in=makematrix(self.num_in,self.num_hidden)
        self.weight_out=makematrix(self.num_hidden,self.num_out)
        #给权重矩阵设置初始值0.1
        for i in range(self.num_in):
            for j in range(self.num_hidden):
                self.weight_in[i][j]=random_num(0.1,0.1)
        for i in range(self.num_hidden):
            for j in range(self.num_out):
                self.weight_out[i][j]=random_num(0.1,0.1)

        #偏差
        for i in range(self.num_hidden):
            self.weight_in[0][i]=0.1
        for i in range(self.num_out):
            self.weight_out[0][i]=0.1

        #动量因子
        self.ci=makematrix(self.num_in,self.num_hidden)
        self.co=makematrix(self.num_hidden,self.num_out)

    #正向传播过程
    def update(self,inputs):
        if len(inputs)!=(self.num_in-1):
            raise ValueError('与输入结点数不符')
        #数据输入
        self.active_in[1:self.num_in]=inputs

        #输入层输入数据乘以输入层权重矩阵得到隐藏层的输入
        self.sum_hidden=np.dot(self.weight_in.T,self.active_in.reshape(-1,1))
        #隐藏层处理数据，得到的结果作为输出层的输入
        self.active_hidden=sigmoid(self.sum_hidden)
        self.active_hidden[0]=-1

        #输出层处理数据
        self.sum_out=np.dot(self.weight_out.T,self.active_hidden.reshape(-1,1))
        self.active_out=sigmoid(self.sum_out)

        #返回输出层结果
        return self.active_out

    #反向传播过程，lr---learn rate 学习速率,targets---目标输出   m--动量系数
    def back_updata(self,targets,lr,m):
        if self.num_out==1:
            targets=[targets]
        if len(targets)!=self.num_out:
            raise ValueError('与输出结点数不符')
        #误差
        error=(1/self.num_out)*np.dot((targets.reshape(-1,1)-self.active_out).T,(targets.reshape(-1,1)-self.active_out))

        #输出层误差信号
        self.error_out=(targets.reshape(-1,1)-self.active_out)*sigmoid_derivatives(self.sum_out)
        #隐藏层误差信号
        self.error_hidden=np.dot(self.weight_out,self.error_out)*sigmoid_derivatives(self.sum_hidden)

        #更新权重矩阵
        #更新隐藏层权重
        self.weight_out=self.weight_out+lr*np.dot(self.error_out,self.active_hidden.reshape(1,-1)).T+m*self.co
        self.co=lr*np.dot(self.error_out,self.active_hidden.reshape(1,-1)).T
        #更新输入层权重
        self.weight_in=self.weight_in+lr*np.dot(self.error_hidden,self.active_in.reshape(1,-1)).T+m*self.ci
        self.ci=lr*np.dot(self.error_hidden,self.active_in.reshape(1,-1)).T

        return error


    #测试
    def test(self,patterns):
        for i in patterns:
            print(i[0:self.num_in-1],"->",self.update(i[0:self.num_in-1]))
        return self.update(i[0:self.num_in-1])

    #权值
    def weights(self):
        print('输入层的权值：',self.weight_in)
        print('输出层的权值',self.weight_out)

    #训练
    def train(self,patterns,itera=100,lr=0.2,m=0.1):
        for i in range(itera):
            error=0.0  #每一次迭代误差重置
            for j in  patterns:
                inputs=j[0:self.num_in-1]
                targets=j[self.num_in-1:]
                self.update(inputs)
                error=error+self.back_updata(targets,lr,m)
            if i%10==0:
                print('########误差 %-.5f ##########第%d次迭代' %(error,i))



#算法检验——预测数据D
# X 输入数据；D 目标数据
X = list(np.arange(-1, 1.1, 0.1))   # -1~1.1 步长0.1增加
D = [-0.96, -0.577, -0.0729, 0.017, -0.641, -0.66, -0.11, 0.1336, -0.201, -0.434, -0.5,
     -0.393, -0.1647, 0.0988, 0.3072,0.396, 0.3449, 0.1816, -0.0312, -0.2183, -0.3201]
A = X + D   # 数据合并 方便处理
patterns = np.array([A] * 2)    # 2*42矩阵
# 创建神经网络，21个输入节点，13个隐藏层节点，21个输出层节点
bp = BP(21, 13, 21)
# 训练神经网络
bp.train(patterns)
# 测试神经网络
d = bp.test(patterns)
# 查阅权重值
bp.weights()


import matplotlib.pyplot as plt
plt.plot(X, D, label="source data")  # D为真实值
plt.plot(X, d, label="predict data")  # d为预测值
plt.show()





