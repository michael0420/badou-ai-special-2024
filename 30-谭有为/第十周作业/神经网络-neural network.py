import numpy as  np
import scipy.special

class Neural_Network:
    #初始化参数：输入结点数、隐藏结点数、输出结点数、学习率
    def __init__(self,num_in,num_hidden,num_out,lr):
        self.num_in=num_in
        self.num_hidden=num_hidden
        self.num_out=num_out
        self.lr=lr
     #初始化权重矩阵,random.rand 随机生成0-1之间的浮点数  -0.5使权重在-0.5-0.5之间
        self.weight_in=np.random.rand(self.num_hidden,self.num_in)-0.5
        self.weight_out=np.random.rand(self.num_out,self.num_hidden)-0.5

    #初始化激活函数sigmoid   lambda--宏定义
        self.active_function=lambda x: scipy.special.expit(x)

        pass

    #训练
    def train(self,inputs_list,target_list):
        #ndmin---用于指定数组的最小维度
        #输入数据处理
        inputs=np.array(inputs_list,ndmin=2).T
        targets=np.array(target_list,ndmin=2).T

        #隐藏层数据处理
        hidden_inputs=np.dot(self.weight_in,inputs)
        hidden_outputs=self.active_function(hidden_inputs)

        #输出层数据处理
        out_inputs=np.dot(self.weight_out,hidden_outputs)
        outputs=self.active_function(out_inputs)

        #误差计算
        errors=targets-outputs
        h_error=np.dot(self.weight_out.T,errors*outputs*(1-outputs))

        #更新权重矩阵
        self.weight_out+=self.lr*np.dot((errors*outputs*(1-outputs)),hidden_outputs.T)
        self.weight_in+=self.lr*np.dot((h_error*hidden_outputs*(1-hidden_outputs)),inputs.T)

        pass

    #推理
    def query(self,inputs):
         hidden_inputs=np.dot(self.weight_in,inputs)
         hidden_outputs=self.active_function(hidden_inputs)
         out_inputs=np.dot(self.weight_out,hidden_outputs)
         outputs=self.active_function(out_inputs)
         return outputs

if __name__=='__main__':
    #参数初始化
    num_in=28*28
    num_hidden=200
    num_out=10
    lr=0.1
    n=Neural_Network(num_in,num_hidden,num_out,lr)

    #读入数据集   open函数中的r表示只读   readlines函数---按行读取文件内容
    training_data_file=open("F:/data/mnist_train.csv",'r')
    training_data_list=training_data_file.readlines()
    training_data_file.close()


    #训练过程
    epchos=10
    for e in range(epchos):
        for data in training_data_list:
            all_vaules=data.split(',')
            #归一化
            # asfarray函数--转换输入为浮点类型的数组  训练数据第一列为标签 后面的列才代表图片像素 每一行代表一张图片
            inputs=(np.asfarray(all_vaules[1:]))/255.0 * 0.99+0.01
            targets=np.zeros(num_out)+0.01
            targets[int(all_vaules[0])]=0.99
            n.train(inputs,targets)

    #测试
    test_data_file=open("F:/data/mnist_test.csv",'r')
    test_data_list=test_data_file.readlines()
    test_data_file.close()

    score=[]
    for data in test_data_list:
        all_vaules=data.split(',')
        true_num=int(all_vaules[0])
        print('该图片对应的数字应为：',true_num)
        inputs=(np.asfarray(all_vaules[1:]))/255.0 * 0.99+0.01
        outputs=n.query(inputs)
        #找到输出中数值最大的 标签
        lable=np.argmax(outputs)
        print('模型认为该数字为：',lable)
        if lable==true_num:
            score.append(1)
        else:
            score.append(0)

    print(score)
    #将列表转换为元组
    score_array=np.asarray(score)
    print(score)
    print("正确率为：",score_array.sum()/score_array.size)





