#随机采样一致性
#内群点与外群点
#ransac是一个求解已知模型的参数的方法
#大致思想是随机选择一组内群数据，存在一个程序，这个程序可以估算这个模型的最佳参数
#具体实现思路：随机选择一组数据作为内群数据，用这些数据拟合一个数学模型(用最小二乘法求k和b)，然后用其他
#点代入这个模型，计算实际点和该模型预测点的差异，如果差异在某个设定的阈值之内，那么
#这个点也是内群点，然后反复多次迭代这个过程，直到选择出内群点最多的数据模型。

#最小二乘法工作原理也是通过求解最佳参数来确定数学模型的，
#但是最小二乘法运算量太大且适用于原数据误差较小的情况（会受到其他噪声点的影响导致模型不能有效拟合数据点）

#ransac的输入
#1.一组内群点，一个数学模型，一些可信的参数

#ransac具体步骤
#1.在数据中随机选几个点作为内群点
#2.通过这几个内群点拟合出一个数学模型
#3.将其他点带入这个数学模型，判断是否为内群点
#4.记录内群点数量
#5.重复以上步骤
#6.比较哪次计算中内群点数量最多，那么这个模型就是最优模型

#需要注意的是ransac在处理不同数据模型时，求解模型参数的方法不同，但是ransac不关注如何求解这个模型的参数

#一开始要随机选几个点？(选小一点较好)
#迭代多少次？

#ransac的优点：可以在包含大量外群的数据中估计参数
#缺点：要求数学模型已知，迭代次数过少可能得到错误结果
import numpy as np
import scipy as sp
import scipy.linalg as sl

class LinearLeastSquareModel_test:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        x, resids, rank, s = sl.lstsq(A, B)  # residues:残差和
        return x  # 返回最小平方和 向量

    def get_error(self, data, k):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = sp.dot(A, k)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point



def random_partiton(n,n_data):
    #随机取内群点
    all_index= np.arange(n_data.shape[0]) #500个样本的索引
    np.random.shuffle(all_index)  #打乱所有索引
    index1 = all_index[:n]
    index2 = all_index[n:]
    return  index1,index2




#先构建数学模型
n_sample = 500
n_inputs = 1
n_outputs = 1
X_exact = 20 * np.random.random(size = (n_sample,n_inputs)) #样本：生成n_sample行n_inputs列数据
perfect_fit = 60 * np.random.normal(size=(n_inputs,n_outputs))  #斜率
Y_exact = np.dot(X_exact, perfect_fit)


#给所有点加入高斯噪声,增加离散趋势
X_noise = X_exact + np.random.normal(size=X_exact.shape)
Y_noise = Y_exact + np.random.normal(size=Y_exact.shape)


#添加局外点
n_outliers = 100
all_index = np.arange(X_noise.shape[0])
np.random.shuffle(all_index)
outliers_index = all_index[:n_outliers]
X_noise[outliers_index] = 20 * np.random.random(size = (n_outliers,n_inputs))
Y_noise[outliers_index] = 50 * np.random.random(size = (n_outliers,n_outputs))

all_data = np.hstack((X_noise,Y_noise))  #形成500，2

input_columns = range(n_inputs)
output_columns = [n_inputs + i for i in range(n_outputs)]

model = LinearLeastSquareModel_test(input_columns, output_columns, debug=False)

linear_fit,resids,rank,s = np.linalg.lstsq(all_data[:,input_columns], all_data[:,1])

#实现ransac
n=50
k = 1000
t = 7e3
d = 300
iterations = 0
bestfit=None
besterr = np.inf
best_inner_index=None
return_all = True
while iterations < k:
    #随机选择一部分样本点作为内群点,innerindex为选出的50个内群点，test_index为
    inner_index , test_index = random_partiton(n,all_data)
    maybeinner = all_data[inner_index]
    test_points = all_data[test_index]

    #使用选出的内群点maybeinner训练模型
    maybemodel = model.fit(maybeinner)
    #使用模型拟合其他数据点计算误差
    test_err = model.get_error(test_points, maybemodel)
    #选择阈值小于t的索引，根据索引在其他数据点中再选择内群点
    also_index = test_index[test_err<t]
    #有了这些点的索引再选择数据点
    also_inners = all_data[also_index]

    #判断内群点的长度是否大于d(d：拟合较好时，需要的样本点最少的个数，当作阈值看待)
    if(len(also_inners) > d):
        betterdata = np.concatenate((maybeinner,also_inners))
        bettermodel = model.fit(betterdata)
        better_errs = model.get_error(betterdata, bettermodel)
        thiserr = np.mean(better_errs)
        if(thiserr < besterr):
            bestfit = bettermodel
            besterr = thiserr
            best_inner_index = np.concatenate((inner_index,also_index))
    iterations += 1


print(bestfit)
print(best_inner_index)

print(all_data[best_inner_index])
#print(all_data.shape)
import pylab
pylab.plot( X_noise,Y_noise, 'k.', label = 'data' )
pylab.plot( all_data[best_inner_index][:,0],all_data[best_inner_index][:,1], 'bx', label = 'data' )


pylab.show()















