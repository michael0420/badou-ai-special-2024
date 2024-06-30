import numpy as np
import scipy as sp
import scipy.linalg as sl


def ransac(data, model, n, k, t, d, debug=False, return_all= False ):
    iterations = 0       #初始化迭代次数为0
    bestfit = None       #用于存储最优拟合模型
    besterr = np.inf     #初始化为正无穷，用于记录最小误差
    best_inlier_idxs = None  #用于储存最佳内点的索引

    while iterations <k:
        #开始迭代，直到达到最大迭代次数‘K’
        maybe_idxs, test_idxs = random_partition(n,data.shape[0]) #maybe_inliers(用于拟合模型的点）和test_points(用于验证模型的点）
        maybe_inliers = data[maybe_idxs, :]#用来拟合模型的数据
        test_points = data[test_idxs]#剩余的若干点为测试点
        maybe_model = model.fit(maybe_inliers)#开始拟合模型
        test_error = model.get_error(test_points,maybe_model)#计算test_points中的点代入maybe_model后的误差
        print('test error =',test_error < t)
        also_idxs = test_idxs[test_error < t]#选择误差小于阈值t的测试点，作为also inliers
        print('also idxs = ',also_idxs)
        also_inliers = data[also_idxs,:]#将这些测试点作为可能得内点
        if debug:#如果debug模式开启
            print('test_err.min()',test_error.min())
            print('test_err.max()',test_error.max())
            print(f'interation:{iterations} ,also inliers:  {len(also_inliers)}')
        if(len(also_inliers)>d):#检查also_inliers的数量是否大于阈值d
            betterdata = np.concatenate((maybe_inliers,also_inliers))#合并作为更好的数据集
            bettermodel = model.fit(betterdata)#使用更好的数据集重新拟合模型
            better_errs = model.get_error(betterdata, bettermodel)#计算betterdata相对于better model的误差
            thiserr = np.mean(better_errs)#计算betterdata的平均误差
            if thiserr < besterr:#如果当前误差小于最小误差，则更新最佳模型拟合、最佳误差和最佳内点索引
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs,also_idxs))
                iterations += 1 #增加迭代次数
    if bestfit is None :#如果未找到合适的模型，则抛出异常
        raise ValueError('did not meet fit aceceptance criteria')
    if return_all:
        return bestfit,{'inliers':best_inlier_idxs}#同时返回内点的索引，返回一个字典，其中包含了内点的索引
    else:
        return bestfit
def random_partition(n,n_data):
    all_idxs = np.arange(n_data)#获取n_data下标索引
    np.random.shuffle(all_idxs)#打乱下标索引
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

class LinearLeastSquare:
    def __init__(self, input_columns, output_columns, debug = False):
        self.input_columns = input_columns#self使得我们能够在类的方法中引用实例属性，而不仅仅是方法的局部变量
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:,i] for i in self.input_columns]).T
        B = np.vstack([data[:,i] for i in self.output_columns]).T
        x, resids, rank, s = sl.lstsq(A, B)
        return x  #返回最小平方和向量

    def get_error(self,data,model):
        A = np.vstack([data[:,i] for i in self.input_columns]).T
        B = np.vstack([data[:,i] for i in self.output_columns]).T
        B_fit = np.dot(A, model)
        err_per_point = np.sum((B-B_fit)**2, axis=1)#逐行计算误差
        return err_per_point

def test():
    n_samples = 500 #样本
    n_inputs = 1 #输入变量，每个样本数据只有1个特征或变量
    n_outputs =1
    A_exact = 20 * np.random.random((n_samples,n_inputs))
    perfect_fit = 60 * np.random.random(size = (n_inputs, n_outputs))#生成模型参数
    B_exact = np.dot(A_exact, perfect_fit)

    #加入高斯噪声，最小二乘法处理
    A_noisy = A_exact + np.random.random(size = A_exact.shape) # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.random(size = B_exact.shape) # 500 * 1行向量,代表Yi

    if 1:
    #加入离群点，帮助我们在测试ransac算法时更好地评估其性能和鲁棒性
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])#获取索引
        np.random.shuffle(all_idxs)
        outlier_idxs = all_idxs[:n_outliers]
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers,n_inputs))
        B_noisy[outlier_idxs] = 50 * np.random.random(size=(n_outliers,n_outputs))

    all_data = np.hstack((A_noisy,B_noisy))
    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for i in range(n_outputs)]
    debug = False
    model = LinearLeastSquare(input_columns, output_columns,debug = debug)#实例化
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:,input_columns], all_data[:,output_columns])

    ransac_fit, ransac_data = ransac(all_data, model, 50, 10, 7e3, 300, debug = debug, return_all= True)

if __name__ == "__main__":
    test()









    '''
    输入：
    data-样本点
    model-假设模型，事前确定
    K-最大迭代次数
    t-阈值：作为判断点满足模型的条件
    n-生成模型所需要的最少样本点
    d-拟合较好时，需要的样本点最少的个数，当做阈值看待

    输出：
    bestfit- 最优拟合解（返回nil，如果未找到）
    '''
