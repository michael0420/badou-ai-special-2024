import numpy as np
import scipy as sp
import scipy.linalg as sl

def ransac(data,model,n,k,t,d,debug = False,return_all = False):
    """
       输入:
           data - 样本点
           model - 假设模型:事先自己确定
           n - 生成模型所需的最少样本点
           k - 最大迭代次数
           t - 阈值:作为判断点满足模型的条件
           d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
       输出:
           bestfit - 最优拟合解（返回nil,如果未找到）

       iterations = 0
       bestfit = nil #后面更新
       besterr = something really large #后期更新besterr = thiserr
       while iterations < k
       {
           maybeinliers = 从样本中随机选取n个,不一定全是局内点,甚至全部为局外点
           maybemodel = n个maybeinliers 拟合出来的可能符合要求的模型
           alsoinliers = emptyset #满足误差要求的样本点,开始置空
           for (每一个不是maybeinliers的样本点)
           {
               if 满足maybemodel即error < t
                   将点加入alsoinliers
           }
           if (alsoinliers样本点数目 > d)
           {
               %有了较好的模型,测试模型符合度
               bettermodel = 利用所有的maybeinliers 和 alsoinliers 重新生成更好的模型
               thiserr = 所有的maybeinliers 和 alsoinliers 样本点的误差度量
               if thiserr < besterr
               {
                   bestfit = bettermodel
                   besterr = thiserr
               }
           }
           iterations++
       }
       return bestfit
       """
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    while iterations <k:
        maybe_idxs,test_idxs = random_partition(n,data.shape[0])
        maybe_inliers = data[maybe_idxs,:]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybe_inliers)
        test_err = model.get_error(test_points,maybemodel)
        also_idxs = test_idxs[test_err<t]
        also_inliers = data[also_idxs,:]
        if (len(also_inliers)>d):
            betterdata = np.concatenate( (maybe_inliers, also_inliers) )


def random_partition(n, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    all_idxs = np.arange(n_data) #获取n_data下标索引
    np.random.shuffle(all_idxs) #打乱下标索引
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2
class LinearLeastSquareModel:
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
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = sp.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point
def test():
    # 生成理想数据
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = 20*np.random.random((n_samples,n_inputs))#随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal( size=(n_inputs, n_outputs) )  #随机线性度，即随机生成一个斜率
    B_exact = sp.dot(A_exact,perfect_fit)

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal( size= A_exact.shape )
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        n_outliners = 100
        all_idxs = np.arange( A_noisy.shape[0] )
        np.random.shuffle(all_idxs)
        outlier_idxs = all_idxs[:n_outliners]
        A_noisy[outlier_idxs] = 20*np.random.random( (n_outliners,n_inputs))
        B_noisy[outlier_idxs] = 50*np.random.normal(size=(n_outliners,n_outputs))

        all_data = np.hstack( (A_noisy,B_noisy))
        input_columns = range(n_inputs)
        output_columns = [n_inputs + i for i in range(n_outputs)]
        debug = False
        model = LinearLeastSquareModel(input_columns, output_columns, debug = debug) #类的实例化:用最小二乘生成已知模型

        linear_fit,resids,rank,s = sl.lstsq(all_data[:,input_columns],all_data[:,output_columns])

        # run RANSAC 算法
        ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)



