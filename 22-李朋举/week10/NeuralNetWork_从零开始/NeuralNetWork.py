import numpy
import scipy.special

# 初始化 - 推理 - 训练
class NeuralNetWork:
    """
    初始化函数，需要在这里设置输入层，中间层和输出层的节点数，这样就能决定网络的形状和大小。
             当然我们不能把这些设置都写死，而是根据输入参数来动态设置网络的形态。
    """
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """
        初始化网络，设置输入层，中间层，和输出层节点数
        """
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        """
        设置学习率
        """
        self.lr = learningrate

        '''
        初始化权重矩阵: 由于权重不一定都是正的，它完全可以是负数，因此我们在初始化时，把所有权重初始化为-0.5到0.5之间
             这里两个权重矩阵，一个是wih(weight-input-hidden)表示输入层和中间层节点间链路权重形成的矩阵，
                            一个是who(weight-hidden-output)表示中间层和输出层间链路权重形成的矩阵 
                                   
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        作用是生成一个正态分布的随机数矩阵:
            `numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))`：
                    `numpy.random.normal()` 函数来生成一个正态分布的随机数矩阵。正态分布是一种常见的概率分布，其平均值为 `0.0`，
                     标准差为 `pow(self.hnodes, -0.5)`。 (`self.hnodes` 的值取负的平方根(到数的平方根)，作为正态分布的标准差)
             最后，`(self.hnodes, self.inodes)` 表示生成的随机数矩阵的大小，即行数为 `self.hnodes`，列数为 `self.inodes`。  
             self.wih => ndarry(200，784) ndtype-float64
             self.who => ndarry(10,200) ndtype-float64
        '''
        # numpy.random.rand(self.hnodes, self.inodes) =》0-1   -0.5 =》 -0.5-0.5
        # self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        # self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        '''
        每个节点执行激活函数，得到的结果将作为信号输出到下一层，我们用sigmoid作为激活函数
        sigmod函数在Python中可以直接调用，我们要做的就是准备好参数, 在init中先把这个函数在初始化函数中设定好，
        '''
        self.activation_function = lambda x: scipy.special.expit(x)
        pass


    """
    训练: 根据输入的训练数据更新节点链路权重 
        第一步是计算输入训练数据，给出网络的计算结果，这点跟我们前面实现的query()功能很像。"正向"
        第二步是将计算结果与正确结果相比对，获取误差，采用误差反向传播法更新网络里的每条链路权重。"反向"
    每一张图片训练一次的过程
    """
    def train(self, inputs_list, targets_list):
        """
                  inputs                hidden_outputs                final_outputs
                  (784,1)                 (200，1)                      (10，1)
                                          zh1/ah1                       zo1/ao1
                            self.wih               self.who
                           (200，784)                     (10,200)

                  hidden_inputs = numpy.dot(self.wih, inputs)         (200，784)  X   (784,1)  = (200，1)
                  final_inputs = numpy.dot(self.who, hidden_outputs)  (10, 200)  X   (200，1)  = (10，1)
        """
        """
        第一步: 计算输入训练数据，给出网络的计算结果，这点跟我们前面实现的query()功能很像。"正向"
           把inputs_list(输入的训练数据), targets_list(训练数据对应的正确结果)转换成numpy支持的二维矩阵, T表示做矩阵的转置
        """
        '''
        将输入列表 `inputs_list` 转换为一个二维数组，并对其进行转置操作：
            1. `numpy.array(inputs_list, ndmin=2)`：
                使用 `numpy.array()` 函数将输入列表 `inputs_list` 转换为一个 `numpy` 数组。`ndmin=2` 参数表示数组的最小维度为 2，
                即输入列表必须至少包含两个元素。如果输入列表的长度小于 2，则会引发 ValueError 异常。
            2. `.T`：这部分代码对上一步生成的数组进行转置操作。转置操作会交换数组的行和列，使得原来的行变为列，原来的列变为行。
            综上所述，这段代码的作用是将输入列表 `inputs_list` 转换为一个二维数组，并对其进行转置操作，以满足后续处理的要求。
        inputs_list-> ndarray(784,)  [0.01  0.01 ...... 0.538  0.99223529, 0.99223529 0.99223529 0.83305882  0.01 ]
        inputs -> ndarray:(784,1)    [[0.01      ], [0.01      ], [0.01      ], .......[0.01      ], [0.86023529], [0.99223529], [0.99223529], [0.16141176], [0.01      ] ]
        '''
        inputs = numpy.array(inputs_list, ndmin=2).T
        '''
        targets_list-> ndarray(10,)  [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        targets-> ndarray:(10,1)     [[0.01], [0.01], [0.01], [0.01], [0.01], [0.99], [0.01], [0.01], [0.01], [0.01]]
        '''
        targets = numpy.array(targets_list, ndmin=2).T
        # 计算信号经过输入层后产生的信号量  矩阵乘(AB的行列数必须满足要求)
        """
        self.wih => ndarry(200，784) dtype-float64     inputs => ndarray:(784,1) dtype-float64   
        hidden_inputs => ndarry(200，1)  [[ 1.33109893e+00], [ 2.90960728e-01], [-8.80236886e-01],......, [-8.08880699e-01], [ 9.41510082e-01]]
        """
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        """
        hidden_outputs =>ndarry(200，1)  [[0.78637499], [0.3250882 ], [0.62135534], ......, [0.78473486], [0.51411813], [0.36048346]]
        """
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层接收来自中间层的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activation_function(final_inputs)

        """
        第二步: 计算误差(用正确结果减去网络的计算结果 targets - final_outputs), 
               是将计算结果与正确结果相比对，获取误差，采用误差反向传播法更新网络里的每条链路权重。"反向"
               
                            【隐藏层->输出层 的权值更新: W5-W8】
                            
                                [整体损失Etotal对W5的偏导值]:
                                
                                      ∂Etotal      ∂Etotal      ∂ao1       ∂zo1
                                      -------  =  --------- * -------- * --------  =  -(t1−ao1) * [ao1∗(1−ao1)]  ∗  aℎ1
                                        ∂W5          ∂ao1       ∂zo1       ∂w5        -------------------------                                                                                                         δo1
                                      用δo1表示输出层单元o1的误差：
                                                   ∂Etotal      ∂ao1        ∂Etotal
                                      δo1      =  --------- * --------  =  --------  = -(t1−ao1) * [ao1∗(1−ao1)]  
                                                     ∂ao1       ∂zo1         ∂zo1      -------------------------                                                   
                                      ∂Etotal       
                                      -------  =   δo1 * aℎ1
                                        ∂W5     
                                                                                          
                                [更新w5的全值]：
                                                          ∂Etotal
                                          w5+ = w5 − η * ----------       
                                                            ∂W5 
                                                                                                                  
                            【输入层->隐藏层 的权值更新: W1-W4】
                              
                                [整体损失Etotal对W1的偏导值]: 
                                        ∂Etotal      ∂Etotal      ∂h1        ∂h1
                                        -------  =  --------- * -------- * --------  
                                          ∂W1          ∂h1        ∂z1        ∂w1
                                                   -------------
                                                   
                                                           ∂Etotal      ∂ai        ∂zi          ∂ahi      ∂zhi
                                                 =   ( ∑  -------- * -------  *  ------ )  *  ------  *  ------ 
                                                             ∂ai        ∂zi        ∂hi          ∂zhi       ∂w1  
                                                   --------------------------------------
                                                   
                                                     ∂EO1   ∂EO2    ∂h1     ∂h1
                                                 = ( ---- + ----) * ---- * -----
                                                     ∂h1    ∂h1     ∂Z1     ∂W1
                                                    -------------
                                                     
                                                     ∂EO1   ∂ao1   ∂zo1     ∂EO2   ∂ao2   ∂zo2     ∂h1    ∂h1
                                                 = [(---- * ---- * ----) + (---- * ---- * ----)] * ---- * ----
                                                     ∂ao1   ∂zo1   ∂zh1     ∂ao2   ∂zo2   ∂zh2     ∂Z1    ∂w1
                                                    --------------------------------------------
                                                      
                                                 = [(ao1-t1)*(ao1*(1-ao1))*w5 + (ao2-t1)*(ao2*(1-ao12)*w7] * ah1*(1-ah1) * i1
                                                   --------------------------------------------------------- 
                                                                        δh1                                                                                           
                                        用δh1表示隐藏层单元h1的误差：
                                                                                                  
                                                           ∂Etotal            ∂Etotal      ∂ai       ∂zi            
                                              δh1     =   ----------  =  ( ∑  -------- * -------  * ------ )    =   [(ao1-t1)*(ao1*(1-ao1))*w5 + (ao2-t1)*(ao2*(1-ao12)*w7]
                                                             ∂h1                ∂ai        ∂zi       ∂h1           
                                                                                                       
                                        ∂Etotal            
                                        -------  =   δh1  * ah1*(1-ah1) * i1
                                          ∂W1                
                                                                               
                                [更新w1的权值]:                
                                                        ∂Etotal
                                        w1+ = w1 − η * ----------  
                                                          ∂W1      
        """
        '''
        targets 标签值   final_outputs 输出值
        '''
        output_errors = targets - final_outputs
        '''
        用δh1表示隐藏层单元h1的误差:
        hidden_errors  [(ao1-t1)*(ao1*(1-ao1))*w5 + (ao2-t1)*(ao2*(1-ao2))*w7]
                                        w5   (ao1-t1)       *       (ao1*(1-ao1))
                                     +  w7   (ao2-t1)       *       (ao2*(1-ao2)）
        '''
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        """
        更新权值: 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上  +=, 后满结果自带了方向
        
                              ∂Etotal
              w5+ = w5 − η * ----------   =  w5 -  η *  -(t1−ao1) * [ao1∗(1−ao1)] ∗ aℎ1   
                                ∂W5                     --------------------------
                                                        δo1 (表示输出层单元o1的误差)             
        """
        #              η    *           ( -(t1−ao1)   * [     ao1     *      (1−ao1)      ]) *         aℎ1
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))
        """
        更新权值: 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上  +=, 后满结果自带了方向
        
                              ∂Etotal
              w1+ = w1 − η * ----------   =  w1 -  η *  [(ao1-t1)*(ao1*(1-ao1))*w5 + (ao2-t1)*(ao2*(1-ao12)*w7] * ah1*(1-ah1) * i1 
                                ∂W1                     -------------------------------------------------------
                                                        δh1 (表示隐藏层单元h1的误差) hidden_errors
        """
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))
        # 训练的过程中更新了权值,这里不需要返回
        pass

    '''
    推理:(正向过程)
        query函数的实现，接收输入数据，通过神经网络的层层计算后，在输出层输出最终结果。
             输入数据要依次经过输入层，中间层，和输出层，并且在每层的节点中还得执行激活函数以便形成对下一层节点的输出信号。
             可以通过矩阵运算把这一系列复杂的运算流程给统一起来。 
             推理过程也定义好了模型,训练推理过程还需要再定义模型
             keras 先将模型定义好(compile,一次),分别再调用训练(fit)测试(evaluate)推理(predict)    
    '''
    def query(self, inputs):
        """
        根据输入数据计算并输出答案
        """
        # 计算中间层从输入层接收到的信号量  y=wx
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算中间层经过激活函数后形成的输出信号量 y=wx+b
        """
        hidden是个一维向量，每个元素对应着中间层某个节点从上一层神经元传过来后的信号量总和.
              于是每个节点就得执行激活函数，得到的结果将作为信号输出到下一层.
        """
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最外层接收到的信号量  y=wx
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算最外层神经元经过激活函数后输出的信号量 y=wx+b
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs



# 初始化网络
'''
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
'''
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读入训练数据
# open函数里的路径根据数据存储的路径来设定
training_data_file = open("dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()  # training_data_list {list: 100} 每个元素都是一个str
training_data_file.close()


'''
加入epocs,设定网络的训练循环次数
在原来网络训练的基础上再加上一层外循环, 但是对于普通电脑而言执行的时间会很长。
epochs 的数值越大，网络被训练的就越精准，但如果超过一个阈值，网络就会引发一个过拟合的问题.
'''
epochs = 5
# 每一张图片训练5次
for e in range(epochs):
    # 把数据依靠','区分，并分别读入 ，并分别读入 每一张图片训练一次
    for record in training_data_list:
        """
        record(str类型)  用逗号分隔  
        all_values:['5', '0', '0', ......, '253', '253',  '0\n']
        """
        all_values = record.split(',')
        # 第一个值对应的是图片的表示的数字，所以我们读取图片数据时要去掉第一个数值 （第一列为标签）
        """
        1.`numpy.asfarray()`主要作用是将输入转换为浮点数类型的`NumPy`数组, 不会改变输入数据的结构或形状，只是改变了数据的类型。
               all_values[1:]  :['0', '0', ......, '253', '253',  '0\n']
               numpy.asfarray(all_values[1:] :['0', '0', ......, '253.0', '253.0',  '0\n']
        2.数据读取完毕后，再对数据格式做些调整，以便输入到神经网络中进行分析，需要做的是将数据“归一化”，也就是把所有数值全部转换到0.01到1.0之间。
               由于表示图片的二维数组中，每个数大小不超过255，通过除以 255.0，可以将整数转换为浮点数，并将其值限制在 [0, 1] 之间。 
               有些数值很小，除以255后会变为0(信息丢失比较严重)，这样会导致链路权重更新出问题。
               所以我们需要把除以255后的结果先乘以0.99，然后再加上0.01，这样所有数据就处于[0.01, 1]之间,
         以避免数据过于接近 0 或 1，从而可能导致数值计算的不稳定性。

                image_array / 255.0  =》 0 - 1     
                image_array / 255.0 * 0.99 => 0 - 0.99
                image_array / 255.0 * 0.99 + 0.01 =》 0 - 1
         inputs-> ndarray(784,)  [0.01  0.01 ...... 0.538  0.99223529, 0.99223529 0.99223529 0.83305882  0.01 ]
        """
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        '''
        设置图片与数值的对应关系, 最外层有10个输出节点
        创建一个全0数组 +0.01 最小值为0.01 , 浮点数更准确(自定义)
              ndarray=(10,)   [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        '''
        targets = numpy.zeros(output_nodes) + 0.01
        """
        all_values[0] 标签列, 将数字变成one-hot
            如：   all_values[0] 是标签 5 
               =》 [0.01 0.01 0.01 0.01 0.01 0.99 0.01 0.01 0.01 0.01]
        """
        targets[int(all_values[0])] = 0.99
        """
        每一张图片训练一次
        根据上述做法，把输入图片给对应的正确数字建立联系，这种联系就可以用于输入到网络中，进行训练。
        由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点。
        这里需要注意的是，中间层的节点我们选择了200个神经元，这个选择是经验值。中间层的节点数没有专门的办法去规定，其数量会根据不同的问题而变化。
        确定中间层神经元节点数最好的办法是实验，不停的选取各种数量，看看那种数量能使得网络的表现最好。
        """
        n.train(inputs, targets)
# test可以更新数据集
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    # 预处理数字图片
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 让网络判断图片对应的数字
    outputs = n.query(inputs)
    # 找到数值最大的神经元对应的编号
    label = numpy.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
'''
该图片对应的数字为: 7
[0.05204647 0.01475931 0.03139924 0.06658749 0.04010123 0.03269264 0.00822899 0.84458636 0.11468105 0.03794511]
 ......
 该图片对应的数字为: 9
[0.01809123 0.06122978 0.02190612 0.07735107 0.23985984 0.02849815 0.0985953  0.05805864 0.06266274 0.08017536]
网络认为图片的数字是： 4
'''

# 计算图片判断的成功率
scores_array = numpy.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)
'''
[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
perfermance =  0.7
'''
