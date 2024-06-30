import numpy as np
import scipy as sp
import scipy.linalg as sl
import pylab

class LinearLeastSquareModel:
    def __init__(self, input_columns, out_columns, debug = False):
        self.input_columns = input_columns
        self.output_columns = out_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T #第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T #第二列Yi-->行Yi
        x, reside, rank, s = sl.lstsq(A, B) #残差和
        # print("A---->", A)
        # print("B---->", B)
        return x
    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  #第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T #第二列Yi-->行Yi
        B_fit = sp.dot(A, model) #计算的y值,B_fit = model.k*A + model.b
        #np.sum(a, axis=0) -------------> 列求和
        #np.sum(b, axis=1) -------------> 行求和
        err_per_point = np.sum((B-B_fit) ** 2, axis=1)
        return err_per_point

def ransac(data, model, n, k, t, d, debug = False, return_all = False):
    """
    :param data: 样本点
    :param model: 假设模型：事先自己确定
    :param n: 生成模型所需要的最少样本点
    :param k: 最大迭代次数
    :param t: 阈值：作为判断点满足模型的条件
    :param d: 拟合较好时，需要的样本点最少的个数，当作阈值看待
    :param debug:
    :param return_all:
    :return: best_fit 最优拟合解
    """

    iterations = 0
    bestfit = None
    besterr = np.inf #设置默认值2
    best_inlier_idxs = None
    while iterations < k: #重复k次
        #随机设置n个点为内群点
        maybe_idxs, test_idxs = random_partition(n, data.shape[0]) #len(maybe_idxs)=50=n, len(test_idxs)=450
        # maybe_inliers为内群点
        maybe_inliers = data[maybe_idxs, :] #获取size(maybe_idxs)行数据(Xi, Yi), maybe_inliers.shape=50*2

        test_points = data[test_idxs] #
        print("test_idxs.shape = ",test_idxs.shape)
        maybemodel = model.fit(maybe_inliers) #拟合模型
        test_err = model.get_error(test_points, maybemodel) #计算误差：平方和最小
        # print("test_err = ", test_err < t)
        also_idxs = test_idxs[test_err < t] #test_err<t，该点满足模型
        print("len(also_idxs) = ", len(also_idxs) )
        also_inliers = data[also_idxs, :]  # also_inliers满足该模型的点
        print("also_inliers.shape = ", also_inliers.shape)
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))

        print("d = ", d)
        if(len(also_inliers) > d): #len(also_inliers) > d, 满足该模型的点需要大于d
            betterdata = np.concatenate((maybe_inliers, also_inliers)) #样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs) #平均误差作为新的误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入
        iterations += 1

    if bestfit is None:
        raise ValueError("id't meet fit acceptance criteria")
    if return_all:
        return bestfit,{'inliers':best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    """
    :param n:
    :param n_data:
    :return:
    """
    all_idxs = np.arange(n_data) #获取n_data下标索引
    np.random.shuffle(all_idxs) #打乱下标
    idxs1 = all_idxs[:n]
    print("len(idxs1) = ", len(idxs1))
    idxs2 = all_idxs[n:]
    print("len(idxs2) = ", len(idxs2))
    return idxs1, idxs2


n_samples = 500 #500个样本
n_inputs = 1  #输入变量个数
n_outputs = 1 #输出变量个数
#(A_exact,B_exact)内群点
A_exact = 20 * np.random.random((n_samples, n_inputs)) #随机生成0-20之间的500个数据：行向量
perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs)) #随机线性度，即随机生成一个斜率
print("perfect_fit---------->",perfect_fit)
B_exact = sp.dot(A_exact, perfect_fit) # y = k * x

#500个内群点加噪声生成(A_noisy,B_noisy)
A_noisy = A_exact + np.random.normal(size=A_exact.shape) #500*1行向量，代表xi
B_noisy = B_exact + np.random.normal(size=B_exact.shape) #500*1行向量，代表yi

#生成离群点(A_noisy,B_noisy)
n_outliers = 100
all_idxs = np.arange(A_noisy.shape[0])#获取索引0-499
np.random.shuffle(all_idxs) #将all_idxs将顺序打乱
outlier_idxs = all_idxs[:n_outliers] #选all_idxs的前100个，0-500的随机局外点
A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

print(A_noisy.shape)
print(B_noisy.shape)

all_data = np.hstack((A_noisy, B_noisy))#形式([Xi,Yi]....) shape:(500,2)500行2列
print(all_data.shape)
input_columns = range(n_inputs) #数组的第一列
print("input_columns--->",input_columns)
output_columns = [n_inputs + i for i in range(n_outputs)]
print("output_columns--->",output_columns)

debug = False


# 类的实例化:用最小二乘生成已知模型
model = LinearLeastSquareModel(input_columns, output_columns, debug = debug)
# 基于最小二乘法原理，通过最小化 残差 的 平方和 来估计线性方程组的最优解
linear_fit, rasids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

ransac_fit, ransac_data =  ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

#A_exact从小到大排列
sort_idxs = np.argsort(A_exact[:,0])
A_col0_sorted = A_exact[sort_idxs] #秩为2的数组


pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")


pylab.plot(A_col0_sorted[:, 0],
           np.dot(A_col0_sorted, ransac_fit)[:, 0],
           label='RANSAC fit')
pylab.plot(A_col0_sorted[:, 0],
           np.dot(A_col0_sorted, perfect_fit)[:, 0],
           label='exact system')
pylab.plot(A_col0_sorted[:, 0],
           np.dot(A_col0_sorted, linear_fit)[:, 0],
           label='linear fit')
pylab.legend()
pylab.show()