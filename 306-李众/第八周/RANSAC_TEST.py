import numpy as np
import scipy as sp
import scipy.linalg as sl


def ransac(data,model,n,t,d,k,debug=False ):
    '''
    输入：
        data：数据，model：拟合模型，n：生成模型需要的最小样本点，t:误差阈值，d:需要样本点的最小个数，k:最大迭代次数
    输出：
        bestfit:拟合参数
    '''
    besterr = t
    bestfit = None
    iterations = 0 #当前迭代次数

    while iterations < k:
        #候选模型样本点 验证候选模型点
        maybe_idx,test_idx = random_partition(n,data.shape[0])
        #取点
        maybe_liners = data[maybe_idx,:]
        test_points = data[test_idx,:]
        #拟合模型
        maybefit = model.fit(maybe_liners)
        #验证其他点的误差
        test_err = model.get_err(test_points, maybefit)
        #取出测试集中误差小于阈值t的点的索引
        #test_err < t 生成一个布尔数组，表示哪些位置的误差小于 t。
        #布尔索引是一种使用布尔数组（布尔值 True 和 False）来选择 NumPy 数组中元素的方法。当布尔数组被用于索引另一个数组时，只有布尔值为 True 的位置会被选中。
        also_idx = test_idx[test_err < t]
        also_liners = data[also_idx,:] #同样拟合模型的数据

        if debug:
            print ('test_err.min()',test_err.min())
            print ('test_err.max()',test_err.max())
            print ('numpy.mean(test_err)',np.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d' %(iterations, len(also_liners)) )
        #如果点数量大于阈值d，合并到测试集中
        if(len(also_liners)>d):
            betterdata = np.concatenate((maybe_liners,also_liners))
            betterfit = model.fit(betterdata)
            better_err = model.get_err(betterdata,betterfit)
            #取平均误差为当前误差
            this_err = np.mean(better_err)
            if(this_err<besterr):
                besterr = this_err
                bestfit = betterfit
        #增加迭代次数
        iterations = iterations + 1

    if bestfit is None:
        #没有找到匹配的拟合参数,抛出异常
        raise ValueError("did't meet fit acceptance criteria")
    else:
        return bestfit
#随机分割函数
def random_partition(n,n_data):
    all_idx = np.arange(n_data)
    np.random.shuffle(all_idx)  # 打乱索引
    maybe_idx = all_idx[:n]
    test_idx = all_idx[n:]

    return maybe_idx,test_idx

#最小二乘法
class LinearLeastSquareModel():
    def __init__(self,input_columns,output_columns):
        self.input_columns = input_columns
        self.output_columns = output_columns

    def fit(self,data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组\
        input = np.vstack([data[:,i] for i in self.input_columns]).T
        output = np.vstack([data[:,i] for i in self.output_columns]).T
        '''
        x, resids, rank, s = sl.lstsq(A, B): 使用 SciPy 的 lstsq 方法进行最小二乘拟合，
        返回拟合参数 x，残差 resids，矩阵的秩 rank 和奇异值 s。
        '''
        x,resids,rank,s = sl.lstsq(input,output)
        #返回参数
        return x

    def get_err(self,data,model):
        #取出数据并转换成行
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = np.dot(A,model)
        err_point = np.sum((B-B_fit)**2,axis=1) #矩阵相减，差值平方，axis=1表示沿着行求和

        return err_point

def test():
    n_samples = 500
    n_input = 1
    n_output = 1

    #输入数据
    in_data =20 * np.random.random((n_samples,n_output))
    #随机斜率
    perfect_fit =60 * np.random.normal(size=(n_input,n_output))#矩阵相乘
    #输出数据
    out_data = np.dot(in_data,perfect_fit)

    #添加噪声
    in_noisy = in_data + np.random.normal(size=in_data.shape)
    out_noisy = out_data + np.random.normal(size=out_data.shape)

    #设置离群点
    n_outliners = 100
    all_idx = np.arange(in_noisy.shape[0])#生成0 ~ in_noisy.shape[0]-1的间隔为1的数组
    np.random.shuffle(all_idx)#打乱顺序
    outliners = all_idx[:n_outliners]
    in_noisy[outliners] = 20 * np.random.random((n_outliners,n_input))
    out_noisy[outliners] = 60 * np.random.normal(size=(n_outliners,n_output))

    #合并数据
    data = np.hstack((in_noisy,out_noisy))
    #数据输入列
    input_columns = range(n_input)
    #数据输出列
    output_columns = [n_input + i for i in range(n_output)]

    #最小二乘法实例化模型+

    model = LinearLeastSquareModel(input_columns,output_columns)

    #最小二乘法拟合的参数
    liner_fit,redies,rank,s = sl.lstsq(data[:,input_columns],data[:,output_columns])
    #ransac拟合的参数
    ransac_fit = ransac(data,model,50,7e3,300,2000,debug=False)

    print("ransac_fit",ransac_fit)
    print("liner_fit",liner_fit)
