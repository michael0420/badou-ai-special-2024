import numpy as np
import scipy as sp
import scipy.linalg as sl

'''
#run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug = debug, return_all = True)

'''


def ransac(data, model, n, max_iteration, t, d, debug=False, return_all=False):
    current_iteration=0

    bestfit = None
    # 最佳误差
    besterrs = np.inf
    # 模型局内点数
    best_inlier_idxs = None
    while current_iteration < max_iteration:
        # 随机选取可能是的内群点
        maybe_inds, test_inds = random_partition(n, data.shape[0])
        # print('test_inds = ', test_inds)
        # 获取size(maybe_idxs)行数据(Xi,Yi)
        maybe_liners = data[maybe_inds, :]
        # 若干行(Xi,Yi)数据点
        test_points = data[test_inds]
        # 拟合模型

        maybemodel = model.fit(maybe_liners)
        test_err = model.get_error(test_points, maybemodel)  # 计算误差:平方和最小
        # print(test_err)
        # 获取判断为内群的点，
        # t - 阈值:作为判断点满足模型的条件
        also_idxs = test_inds[test_err < t]
        # print(also_idxs)
        # 获取判断为内群的数量
        also_inliers = data[also_idxs, :]
        # print(len(also_inliers))
        #
        # d - 拟合较好时, 需要的样本点最少的个数, 当做阈值看待
        # 当内群数量大于d时，表明此模型拟合比较好，对此模型加入所有符合的内群点进行更新
        if (len(also_inliers) > d):
            # 将所有内群样本集合起来生成新模型，并计算误差
            betterdata = np.concatenate((maybe_liners, also_inliers))
            bettermodel = model.fit(betterdata)
            bettererrs = model.get_error(betterdata, bettermodel)
            # 平均误差作为新的误差
            xingerrs = np.mean(bettererrs)
            if xingerrs < besterrs:
                bestfit = bettermodel
                besterrs = xingerrs
                best_inlier_idxs = np.concatenate((maybe_inds, also_idxs))  # 更新局内点,将新点加入
        current_iteration += 1
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
        B_fit = np.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point

def test():
    # 编写测试样本
    sample = 300
    inputs = 1
    outputs = 1
    # 随机生成0-30之间的300个数据：行向量
    x_vectors = 30 * np.random.random((sample, inputs))
    # 随机线性度，即随机生成一个斜率
    k_fit = 50 * np.random.normal(size=(inputs, outputs))
    # y=x*k
    y_vectors = np.dot(x_vectors, k_fit)
    # print(y_vectors)

    # 加入高斯噪声,最小二乘能很好的处理  产生新的数据用来测试
    # 300 * 1行向量,代表Xi
    x_noise = x_vectors + np.random.normal(size=x_vectors.shape)
    # print(x_noise[0])
    # 300 * 1行向量,代表Yi
    y_noise = y_vectors + np.random.normal(size=y_vectors.shape)

    # 添加“局外点”也就是离群点
    outliners = 50
    # 获取数据的索引
    all_index = np.arange(x_noise.shape[0])
    # print(all_index)
    # 将all_idxs索引打乱
    np.random.shuffle(all_index)
    # 50个0-300的随机局外点
    out_inds = all_index[:outliners]
    # print(out_inds)
    x_noise[out_inds] = 20 * np.random.random((outliners, inputs))
    y_noise[out_inds] = 50 * np.random.normal(size=(outliners, outputs))
    # 形式([Xi,Yi]....) shape:(500,2)500行2列
    # 合并两个数据
    all_data = np.hstack((x_noise, y_noise))
    # setup model
    input_columns = range(inputs)  # 数组的第一列x:0
    # print(input_columns)
    output_columns = [inputs + i for i in range(outputs)]  # 数组最后一列y:1
    # print(output_columns)
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型
    # print(model)

    # 准备画图所用数据
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])
    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 40, 500, 7e3, 200, debug=debug, return_all=True)
    # print(type(ransac_data))
    if 1:
        import pylab
        sort_idxs = np.argsort(x_vectors[:, 0])
        A_col0_sorted = x_vectors[sort_idxs]  # 秩为2的数组
        if 1:
            pylab.plot(x_noise[:, 0], y_noise[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(x_noise[ransac_data['inliers'], 0], y_noise[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
        else:
            pylab.plot(x_noise[non_outlier_idxs, 0], y_noise[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(x_noise[outlier_idxs, 0], y_noise[outlier_idxs, 0], 'r.', label='outlier data')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, k_fit)[:, 0],
                   label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')

        pylab.legend()
        pylab.show()

if __name__ == "__main__":
    test()

