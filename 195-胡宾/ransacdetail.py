import numpy as np
import scipy as sp
import scipy.linalg as sl


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
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
    besterr = np.inf  # 设置默认值
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        print('test_idxs = ', test_idxs)
        maybe_inliers = data[maybe_idxs, :]  # 获取size(maybe_idxs)行数据(Xi,Yi)
        test_points = data[test_idxs]  # 若干行(Xi,Yi)数据点
        maybemodel = model.fit(maybe_inliers)  # 拟合模型
        test_err = model.get_error(test_points, maybemodel)  # 计算误差:平方和最小
        print('test_err = ', test_err < t)
        also_idxs = test_idxs[test_err < t]
        print('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs, :]
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', numpy.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        # if len(also_inliers > d):
        print('d = ', d)
        if (len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    """return n random rows of data and the other len(data) - n rows"""
    all_idxs = np.arange(n_data)  # 获取n_data下标索引
    np.random.shuffle(all_idxs)  # 打乱下标索引
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


def ff():
    # 准备样板数据
    sample_num = 618
    input_num = 1
    out_put = 1
    row_vector = 20 * np.random.random((sample_num, input_num))
    # 生成高斯正态分布的数组
    k = 60 * np.random.normal(size=(input_num, out_put))

    hi = np.dot(row_vector, k)

    ja_zao_sheng1 = row_vector + np.random.normal(size=row_vector.shape)

    ja_zao_sheng2 = hi + np.random.normal(size=hi.shape)
    if True:
        # 添加局外点
        outlier_point = 100
        index = np.arange(ja_zao_sheng1.shape[0])
        np.random.shuffle(index)
        out_index = index[:outlier_point]
        ja_zao_sheng1[out_index] = 20 * np.random.random((outlier_point, input_num))
        ja_zao_sheng2[out_index] = 50 * np.random.normal(size=(outlier_point, out_put))

    hstack = np.hstack((ja_zao_sheng1, ja_zao_sheng2))
    one_columns = range(input_num)
    last_columns = [input_num + i for i in range(out_put)]  # 数组最后一列
    flag = False
    shuxue_monxing = LinearLeastSquareModel(one_columns, last_columns, debug=flag)
    linear_fit, resids, rank, s = sp.linalg.lstsq(hstack[:, one_columns], hstack[:, last_columns])
    # 执行ransac
    fit, data = ransac(hstack, shuxue_monxing, 50, 1000, 7e3, 300, debug=flag, return_all=True)
    print(fit)
    print(data)
    # print(data.shape[:, ])
    if True:
        import pylab
        n_index = np.argsort(row_vector[:, 0])
        index_ = row_vector[n_index]
        if 1:
            pylab.plot(ja_zao_sheng1[:, 0], ja_zao_sheng2[:, 0], "k.", label="data")
            pylab.plot(ja_zao_sheng1[data['inliers'], 0], ja_zao_sheng2[data['inliers'], 0], "bx", label="RANSAC data")
        else:
            pylab.plot(ja_zao_sheng1[non_outlier_idxs, 0], ja_zao_sheng2[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(ja_zao_sheng1[outlier_idxs, 0], ja_zao_sheng2[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(index_[:, 0],
                   np.dot(index_, fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(index_[:, 0],
                   np.dot(index_, k)[:, 0],
                   label='exact system')
        pylab.plot(index_[:, 0],
                   np.dot(index_, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()


if __name__ == "__main__":
    ff()
