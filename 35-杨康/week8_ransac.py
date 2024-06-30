import numpy as np
from scipy.linalg import lstsq
import pylab

def ransac(data, model, n, k, t, d, debug=False, return_all = False):
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
    """
    iteration = 0
    bestfit = None
    besterr = np.inf
    best_idx = None
    while iteration<k:
        maybe_idx, test_idx = random_parttion(n,data)
        maybe_data = data[maybe_idx]
        test_data = data[test_idx]
        maybefit = model.fit(maybe_data)
        test_err = model.get_err(test_data,maybefit)
        print('test_err:',test_err<t)
        also_idx = test_idx[test_err<t]
        print('also_idx',also_idx)
        also_data = data[also_idx,:]
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iteration, len(also_data)))
        print('d=',d)
        if len(also_data)>d:
            betterdata = np.concatenate((maybe_data,also_data))
            betterfit = model.fit(betterdata)
            better_err = model.get_err(betterdata,betterfit)
            thiserr = np.mean(better_err)
            if thiserr < besterr:
                besterr = thiserr
                bestfit = betterfit
                best_idx = np.concatenate((maybe_idx,also_idx))
        iteration+=1
    if bestfit == None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit,best_idx
    else:
        return bestfit

class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns, debug):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = data[:,self.input_columns]
        B = data[:,self.output_columns]
        # A = np.vstack([data[:, i] for i in self.input_columns]).T# 第一列Xi-->行Xi
        # B = np.vstack([data[:, i] for i in self.output_columns]).T # 第二列Yi-->行Yi
        x, resids, rank, s = lstsq(A, B)  # residues:残差和
        return x  # 返回最小平方和向量
    def get_err(self, data, model):
        A = data[:, self.input_columns]
        B = data[:, self.output_columns]
        B_fit = np.dot(A, model)   #Y = model.k*X + model.b
        err = np.sum((B_fit-B)**2, axis=1)   #计算误差平方和
        return err

def random_parttion(n,ndata):
    all_idxs = np.arange(ndata.shape[0])
    np.random.shuffle(all_idxs)
    idx1 = all_idxs[:n]
    idx2 = all_idxs[n:]
    return idx1,idx2


def test():
    n = 500
    n_X = 1
    n_Y = 1
    X_exact = 20*np.random.random((n, n_X)) #X
    perfect_fit = 60*np.random.normal(size=(n_X, n_Y)) #随机生成一个斜率k
    Y_exact = np.dot(X_exact, perfect_fit) #Y = kX
    #增加高斯噪声
    X_noise = X_exact + np.random.normal(size=X_exact.shape)
    Y_noise = Y_exact + np.random.normal(size=Y_exact.shape)
    #添加局外点
    n_outliers = 100
    all_idx = np.arange(X_noise.shape[0])
    np.random.shuffle(all_idx)
    outlier_idx = all_idx[:n_outliers]
    X_noise[outlier_idx] = 20*np.random.random((n_outliers, n_X))
    Y_noise[outlier_idx] = 50*np.random.normal(size=(n_outliers, n_Y))
    all_data = np.hstack((X_noise,Y_noise))
    print(all_data.shape)
    input_columns = range(n_X)  # 数组的第一列x:0
    output_columns = [n_X + i for i in range(n_Y)]  # 数组最后一列y:1
    print(input_columns,output_columns)
    linear_fit,resids,rank,s = lstsq(X_noise,Y_noise) #lstsq(all_data[:,input_columns], all_data[:,output_columns])
    print(linear_fit)
    debug = True
    model = LinearLeastSquareModel(input_columns,output_columns,debug)
    ransac_fit,ransac_idx = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    #画图
    sorted_col0_idx = np.argsort(X_exact[:, 0])
    pylab.plot(X_noise[:, 0], Y_noise[:, 0], 'k.', label='data')
    pylab.plot(all_data[ransac_idx, 0], all_data[ransac_idx, 1], 'bx', label="RANSAC data")
    pylab.plot(X_exact[:, 0], Y_exact[:, 0], label='exact system')
    pylab.plot(X_exact[sorted_col0_idx], np.dot(X_exact[sorted_col0_idx], linear_fit), label='linear')
    pylab.plot(X_exact[sorted_col0_idx], np.dot(X_exact[sorted_col0_idx], ransac_fit), label='ransac')
    pylab.legend()
    pylab.show()

if __name__ == '__main__':
    test()





