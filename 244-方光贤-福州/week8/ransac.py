import numpy as np
from scipy import linalg as sl
import pylab

def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    RANSAC (Random Sample Consensus) 算法实现

    参数:
    data: 包含输入和输出特征的数据数组
    model: 线性最小二乘模型类实例
    n: 用于拟合模型的随机样本点数量
    k: 最大迭代次数
    t: 内点误差阈值
    d: 所需内点数量的最小阈值
    debug: 是否打印调试信息
    return_all: 是否返回所有内点索引

    返回:
    bestfit: 最优拟合模型参数
    (可选) 字典，包含'inliers'键，其值为内点索引数组

    """
    iterations = 0
    bestfit = None
    besterr = np.inf  # 设置默认误差为无限大
    best_inlier_idxs = None

    # 开始RANSAC迭代
    while iterations < k:
        # 随机划分数据索引
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])

        # 获取可能的内点样本数据
        maybe_inliers = data[maybe_idxs, :]

        # 获取测试点数据
        test_points = data[test_idxs]

        # 使用可能的内点样本拟合模型
        maybemodel = model.fit(maybe_inliers)

        # 计算测试点的误差
        test_err = model.get_error(test_points, maybemodel)

        # 找出误差小于阈值的测试点索引（即内点）
        also_idxs = test_idxs[test_err < t]

        also_inliers = data[also_idxs,:]

        # 调试输出
        if debug:
            print('当前迭代:', iterations)
            print('test_idxs = ', test_idxs)
            print('test_err < t 的索引:', also_idxs)
            print('最小误差:', test_err.min())
            print('最大误差:', test_err.max())
            print('平均误差:', numpy.mean(test_err))
            print('当前内点数量:', len(also_idxs))

            # 如果当前内点数量大于最小阈值d
        if len(also_inliers) > d:
            # 合并可能的内点和当前找到的内点
            betterdata = np.concatenate((maybe_inliers, data[also_idxs, :]))

            # 使用合并后的数据拟合更优的模型
            bettermodel = model.fit(betterdata)

            # 计算更优模型的误差
            better_errs = model.get_error(betterdata, bettermodel)

            # 计算平均误差
            thiserr = np.mean(better_errs)

            # 如果更优模型的误差更小，则更新最优模型和最优内点索引
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))

                # 迭代次数加一
        iterations += 1

        # 如果没有找到满足条件的模型，则抛出异常
    if bestfit is None:
        raise ValueError("没有满足拟合接受标准的模型")

        # 根据return_all的值决定返回内容
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit

# 函数：将数据集索引随机划分为两部分
def random_partition(n, n_data):
    # 获取n_data个下标索引，形成一个数组
    all_idxs = np.arange(n_data)  # 获取n_data下标索引
    # 打乱下标索引数组
    np.random.shuffle(all_idxs)  # 打乱下标索引
    # 划分索引数组为两部分，第一部分包含前n个索引，第二部分包含剩余的索引
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    # 返回划分后的两部分索引
    return idxs1, idxs2


# 类：表示线性最小二乘模型
class LinearLeastSquareModel:
    # 最小二乘求线性解，用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        # 输入列索引和输出列索引
        self.input_columns = input_columns
        self.output_columns = output_columns
        # 调试标志
        self.debug = debug

        # 使用数据拟合模型

    def fit(self, data):
        # 从数据集中提取输入和输出列的数据
        # A表示输入矩阵（每行是一个样本的输入特征）
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        # B表示输出矩阵（每行是一个样本的输出值）
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        # 使用最小二乘法求解线性方程组 Ax = B
        x, resids, rank, s = sl.lstsq(A, B)  # residues: 残差和
        # 返回求解得到的系数向量x
        return x  # 返回最小平方和向量

    # 计算数据点与模型之间的误差
    def get_error(self, data, model):
        # 从数据集中提取输入列的数据
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        # 假设输出只有一列，直接取这一列
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        # 使用模型系数和输入矩阵A计算拟合的输出值B_fit
        B_fit = np.dot(A, model)  # 计算的y值, B_fit = model.k*A + model.b（如果model是斜率和截距）
        # 计算每个数据点的平方误差（每行一个数据点）
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        # 返回每个数据点的平方误差数组
        return err_per_point


def test():
    # 生成理想数据
    sample = 500  # 样本个数
    input = 1  # 输入变量个数
    output = 1  # 输出变量个数
    A_exact = 20 * np.random.random((sample, input))  # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(input, output))  # 随机线性度，即随机生成一个斜率
    B_exact = np.dot(A_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        # 添加"局外点"
        outlier = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idxs)  # 将all_idxs打乱
        outlier_idxs = all_idxs[:outlier]  # 100个0-500的随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((outlier, input))  # 加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(outlier, output))  # 加入噪声和局外点的Yi
    # setup model
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(input)  # 数组的第一列x:0
    output_columns = [input + i for i in range(output)]  # 数组最后一列y:1
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型

    linear_fit, resid, rank, s = sl.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    # 使用numpy的argsort函数对数组A_exact的第一列进行排序，并获取排序的索引
    sort_idxs = np.argsort(A_exact[:, 0])
    # 使用排序的索引对A_exact进行重排，得到一个按照第一列值排序后的数组A_col0_sorted
    A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组（这里假设A_exact是二维数组）

    # 又一个始终为真的条件判断（同样可以省略），这里可能是为了嵌套或清晰度

    # 使用pylab的plot函数绘制A_noisy和B_noisy的第一列数据点，使用黑色'.'作为标记，并添加标签'data'
    pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
    # 使用RANSAC方法选择的inliers数据点（可能是无噪声或噪声较少的数据点）绘制蓝色'x'标记，并添加标签"RANSAC data"
    pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                label="RANSAC data")

    # 绘制通过RANSAC算法拟合的曲线。ransac_fit可能是通过RANSAC算法得到的模型参数
    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit')

    # 绘制通过某种方法（可能是最小二乘法）得到的精确系统拟合曲线。perfect_fit可能是精确拟合的模型参数
    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:, 0], label='exact system')

    # 绘制线性拟合的曲线。linear_fit可能是通过线性最小二乘法等方法得到的模型参数
    pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit)[:, 0], label='linear fit')

    # 添加图例，显示之前设置的标签
    pylab.legend()
    # 显示图形
    pylab.show()

if __name__ == "__main__":
    test()
