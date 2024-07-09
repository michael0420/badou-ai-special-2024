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
    """
    # 初始化
    iterations = 0             # 迭代计数，一开始记为0
    bestfit = None             # 最佳匹配模型，一开始无
    besterr = np.inf           # 最小误差，一开始设置为最大值（np.inf 表示+∞，没有确切的数值,浮点数）
    best_inlier_idxs = None    # 最佳内群，一开始无
    while iterations < k:      # 小于最大迭代次数k就抑制循环
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])     # 返回的maybe_idxs是内群数据行索引，test_idxs是非内群数据行索引
        print ('test_idxs = ', test_idxs)
        maybe_inliers = data[maybe_idxs, :]                            # 获取size(maybe_idxs)行数据(Xi,Yi)
        test_points = data[test_idxs]                                  # 若干行(Xi,Yi)数据点
        maybemodel = model.fit(maybe_inliers)                          # 用内群数据拟合一个模型（函数）
        test_err = model.get_error(test_points, maybemodel)            # 用拟合的模型计算测试数据点的误差:平方和最小
        print('test_err = ', test_err <t)                              # 打印测试误差是否小于某个阈值t
        also_idxs = test_idxs[test_err < t]                            # 根据测试误差小于阈值的条件，找到对应的数据行索引
        print ('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs, :]                              # 从数据集中获取通过误差检验的内群数据行
        if debug:                                                      # 如果 debug 为真，则执行后续缩进块内的代码，用于调试目的
            print ('test_err.min()', test_err.min())
            print ('test_err.max()', test_err.max())
            print ('numpy.mean(test_err)', np.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d' %(iterations, len(also_inliers)) ) #格式化字符串，包含迭代次数 iterations 和 also_inliers 的长度
        print('d = ', d)

        # 关键的来了
        if (len(also_inliers) > d):
            betterdata = np.concatenate( (maybe_inliers, also_inliers) ) # 样本连接
            bettermodel = model.fit(betterdata)                          # 用连接后的数据集 betterdata 对模型 model 进行重新拟合，得到更好的模型 bettermodel
            better_errs = model.get_error(betterdata, bettermodel)       # 使用更好模型 bettermodel 对 betterdata 的预测误差，存储在 better_errs 中
            thiserr = np.mean(better_errs)                               # 平均误差作为新的误差
            if thiserr < besterr:                                        # besterr初始是+∞
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate( (maybe_idxs, also_idxs) ) #更新最佳局内点索引为两部分可能局内点 maybe_idxs 和确认局内点 also_idxs 的连接
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:                                                       # 如果 return_all 为真，会返回最佳模型以及其他相关信息
        return bestfit, {'inliers': best_inlier_idxs}                    # 最佳模型 bestfit，第二个元素是一个字典，其中包含键为 'inliers'，对应的值是最佳局内点的索引 best_inlier_idxs
    else:
        return bestfit                                                   # 如果 return_all 为假，即不需要返回所有信息，那么只返回最佳模型 bestfit



def random_partition(n, n_data):       # 随机分割给定数量的数据行
    """n: 表示要保留为内群的随机数据行数，n_data: 表示总共的数据行数"""
    all_idxs = np.arange(n_data)       # 把数据变成numpy数组，获取n_data下标索引
    np.random.shuffle(all_idxs)        # 打乱下标索引
    idxs1 = all_idxs[:n]               # 取出前n个索引作为idxs1，这些数据行将被保留为内群（0-n）
    idxs2 = all_idxs[n:]               # 取剩余的索引作为idxs2，这些数据行将不属于内群，而是其他数据行(n到最后)
    return idxs1, idxs2

class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):                                            # fit 方法通过最小二乘法拟合输入数据
        A = np.vstack([data[:, i] for i in self.input_columns]).T   # 垂直堆叠在一起，.T 每一行代表一个样本的输入特征
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi（列变成行）
        x, resids, rank, s = sl.lstsq(A, B)                         # 利用 np.linalg.lstsq() 函数进行最小二乘拟合。它会根据输入特征 A 和对应的输出值 B，返回拟合的系数 x，残差 resids，秩 rank 和奇异值 s
        return x                                                    # 返回最小平方和向量x ，x是拟合得到的系数，它表示了输入特征和输出值之间的线性关系

    def get_error(self, data, model):                               # 计算数据集中每个样本的预测误差
        A = np.vstack([data[:, i] for i in self.input_columns]).T   # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = sp.dot(A, model)                                    # 用model计算的y值,B_fit = model.k*A ，这里假设模型 model 是线性的，所以是简单的矩阵乘法运算
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)            # 实际输出值 B 与用模型预测的输出值 B_fit 之间的平方差,np.sum() 函数按指定轴（axis=1，表示按行）对平方差进行求和，得到每个样本的总预测误差
        return err_per_point


def test():
    # 生成理想数据，自己构造数据
    n_samples = 500  # 样本个数
    n_inputs = 1     # 输入变量个数
    n_outputs = 1    # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))           # 随机生成0-20之间的500个输入数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    B_exact = sp.dot(A_exact, perfect_fit)                           # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi，每个样本（行）都添加了一个高斯噪声
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        # 添加"局外点"
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])   # 获取索引0-499
        np.random.shuffle(all_idxs)              # 将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]     # 在打乱的索引数组中选取了前n_outliers个元素，这些元素将用作局外点的索引
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))        # 加入噪声和局外点的Xi，为选择的局外点位置赋予了一些随机生成的值
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi

    all_data = np.hstack((A_noisy, B_noisy))  # 写成形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)           # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1，从n_inputs到n_inputs+n_outputs-1。这表示新数据集中的从第n_inputs列到最后一列是输出数据的列
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型，用作比较

    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)
    # 它将数据集all_data和之前创建的模型model作为输入，然后设置了RANSAC算法的参数，如迭代次数50、最小样本数1000、最大残差7e3、内点阈值300等。此外，通过设置return_all为True，函数将返回RANSAC拟合后的模型参数ransac_fit和内点数据ransac_data


    # 画图，绘制散点图和拟合曲线
    if 1:                                       # 这是一个条件语句，始终为真，用来控制后续代码块的执行
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])   # 对A_exact矩阵的第一列进行排序，返回排序后的索引值
        A_col0_sorted = A_exact[sort_idxs]      # 秩为2的数组，使用排序后的索引值对A_exact矩阵进行重新排序

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 绘制A_noisy和B_noisy的第一列数据的散点图，用黑色点表示，设置标签为'data'
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")           # 根据RANSAC算法得到的内点信息绘制散点图，用蓝色×表示，设置标签为'RANSAC data'
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')  # 绘制A_noisy和B_noisy中非异常值索引对应的数据点的散点图，用黑色点表示，设置标签为'noisy data'
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')        # 绘制A_noisy和B_noisy中异常值索引对应的数据点的散点图，用红色点表示，设置标签为'outlier data'

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')         # 绘制RANSAC算法拟合的曲线，将排序后的A_exact矩阵和RANSAC拟合系数进行矩阵乘法得到拟合结果
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')       # 绘制完美系统的曲线，将排序后的A_exact矩阵和完美拟合系数进行矩阵乘法得到拟合结果
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')         # 绘制线性拟合的曲线，将排序后的A_exact矩阵和线性拟合系数进行矩阵乘法得到拟合结果
        pylab.legend()  # 显示图例
        pylab.show()    # 显示绘制的图形


if __name__ == "__main__":
    test()