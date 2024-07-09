import numpy as np
import scipy as sp
import scipy.linalg as sl


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点  50
        k - 最大迭代次数  1000
        t - 阈值:作为判断点满足模型的条件  7e3
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待  300
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
    iterations = 0  # 迭代计数
    bestfit = None  # 最佳匹配
    besterr = np.inf  # 最佳误差,设置默认值  特殊值-它表示无穷大。
    best_inlier_idxs = None  # 最佳内群的idxs
    while iterations < k:  # k最大迭代次数
        '''
        1.在数据中随机选择一些点设定为内群
        在每次迭代中，进行随机分区并选择数据点：
           使用了自定义`random_partition`函数来对数据进行随机分区： 返回两个列表：`maybe_idxs`(n=50)和`test_idxs`(500-50)，分别表示可能的内群索引和测试点索引。
           具体来说，`random_partition`函数会随机选择一部分数据点作为可能的内群，然后将剩余的数据点作为测试点。
                    在每次迭代中，这些随机选择的索引会被重新生成，以确保每次迭代都使用不同的数据点进行拟合和测试，从而提高算法的鲁棒性和泛化能力。
        '''
        # 随机选取 50个内群点索引 [150 410 485 ...]  450个离群点索引 [272 190 104 ...]    data(500,2)
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])  # 可能的内群索引和测试点索引
        print('test_idxs = ', test_idxs)
        maybe_inliers = data[maybe_idxs, :]
        # 获取size(maybe_idxs)行数据(Xi,Yi)  50个内群点 [[ 1.35171458e+01  1.20891452e+02], [ 1.49500689e+01  1.22343406e+02], [ 5.42782452e+00  4.12638319e+01]...]
        test_points = data[test_idxs]
        # 若干行(Xi,Yi)数据点  450个离群点 (500,2) [[ 8.79899064e+00 -2.32684054e+01], [ 1.39586115e+01  1.22994147e+02], [ 1.26526117e+01 -3.44334034e+01]...]
        '''
        2.计算适合内群的模型   maybemodel -> [[7.27365696]]
        使用模型对象`model`对可能的内群数据`maybe_inliers`进行拟合，得到一个新的模型对象`maybemodel`。
        通过拟合可能的内群数据，可以得到一个更准确的模型，然后可以使用这个模型来预测其他数据点的值，或者进行其他相关的分析和处理。
        '''
        maybemodel = model.fit(maybe_inliers)  # 拟合模型
        '''
        3.把其他刚才没选到的点带入到刚才建立的模型中，计算是否为内群。 test_err -> (450,)  [7.61592111e+03 4.60703077e+02 1.59931839e+04 ...]
        计算模型在测试点上的误差。其中，`model`是模型对象，`test_points`是测试点的数据，`maybe_model`是可能的模型。
        `get_error`是模型对象的一个方法，用于计算模型在给定数据上的误差。通过计算模型在测试点上的误差，可以评估模型的性能和精度，
                                     并根据误差的大小来调整模型的参数或选择更适合的模型。
        '''
        test_err = model.get_error(test_points, maybemodel)  # 计算误差:平方和最小
        print('test_err = ', test_err < t)
        ''' 
        `also_idxs = test_idxs[test_err < t]`的意思是，将测试误差小于阈值`t`的测试点索引存储在变量`also_idxs`中。
         这意味着，这些测试点的误差较小，因此可能更适合用于进一步的分析和处理。
         目的是筛选出误差较小的测试点，以便进一步分析这些点的特征，或者用于模型的改进和优化。 also_idxs -> (423,) [190 385 223 ...]
        '''
        also_idxs = test_idxs[test_err < t]
        print('also_idxs = ', also_idxs)
        '''
        从数据矩阵`data`中选择索引为`also_idxs`的行，并将这些行存储在新的矩阵`also_inliers`中。
        具体来说，`data`是一个二维矩阵，它包含了所有的数据点。`also_idxs`是一个一维数组，它包含了测试误差小于阈值`t`的测试点的索引。
        通过使用索引`also_idxs`，代码从`data`中选择了对应的行，并将这些行存储在新的矩阵`also_inliers`中。  
        目的是从原始数据中选择符合条件的测试点，并将这些测试点的行数据存储在`also_inliers`中，以便进行进一步的分析和处理。
        also_inliers -> (413,2) [[ 1.39586115e+01  1.22994147e+02], [ 5.26324827e+00  5.10124649e+01], [ 6.38467532e+00  5.06180268e+01]...]
        '''
        also_inliers = data[also_idxs, :]
        if debug:
            print('test_err.min()', test_err.min())  # 0.007502442446006568
            print('test_err.max()', test_err.max())  # 74936.76344055624
            print('numpy.mean(test_err)', numpy.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))  # 0  413
        # if len(also_inliers > d):
        print('d = ', d)
        '''
        4. 记下内群数量  
        '''
        if (len(also_inliers) > d):  # 在>d的情况下才进入比较的队列  413 > 300
            '''
            将矩阵maybe_inliers和also_inliers沿着行方向进行连接，得到一个新的矩阵betterdata。连接后的矩阵betterdata的行数是两个原始矩阵的行数之和，列数与原始矩阵的列数相同。
            betterdata -> (463,2) [[ 1.35171458e+01  1.20891452e+02], [ 1.49500689e+01  1.22343406e+02], [ 5.42782452e+00  4.12638319e+01] ...]
            '''
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接  内群点 =  50个内群点 + 450个离群点中误差<阈值t的离群点

            '''
            一次比较不错的结果, 重新计算 最佳模型和最佳误差 
            bettermodel -> [[8.31612951]]  better_errs -> (463,2)[7.19293434e+01 3.93349098e+00 1.50129888e+01 ...] 
            '''
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)

            '''
            6. 比较那次计算中内群数量最多，内群数量最多的那次所建的模型就是我们所要求的解
            计算新的误差值`thiserr`，它是连接后的更好数据`betterdata`的误差的平均值。
            具体来说，`np.mean(better_errs)`表示使用`numpy`库中的`mean`函数计算更好数据`betterdata`的误差值`better_errs`的平均值，并将结果赋值给变量`thiserr`。
            '''
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差   float64 742.1489530861047   besterr float inf
            if thiserr < besterr:
                bestfit = bettermodel  # [[8.31612951]]
                besterr = thiserr  # 742.1489530861047
                # 最佳匹配时的 内群点的个数  （463，2） [150 410 485 ...]
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


''' 最小二乘法'''


class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    '''
    函数来执行线性最小二乘拟合。这个函数可以找到一个最佳的线性模型
    x->返回最小平方和向量(在线性方程中代表自变量x的变化率), resids->残差, rank->矩阵的秩, s->均方误差 残差的平方和的平均值。
    '''

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi    按垂直方向（行顺序）堆叠数组构成一个新的数组
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi   按水平方向（列顺序）堆叠数组构成一个新的数组
        # （A,B）将输入数据和输出数据按行堆叠，形成一个二维数组A和B，以便进行最小二乘法拟合
        x, resids, rank, s = sl.lstsq(A, B)  # residues:残差和
        return x  # 返回最小平方和向量

    '''
    接受两个参数`data`和`model`，并返回一个新的数组`err_per_point`。
    首先，代码使用列表推导式将输入数据`data`按照输入列`self.input_columns`和输出列`self.output_columns`进行分割，
    然后使用`np.vstack`函数将分割后的输入数据按行堆叠，形成一个新的数组`A`。接着，使用`np.vstack`函数将分割后的输出数据按行堆叠，形成一个新的数组`B`。
    然后，代码调用`sp.dot`函数对输入数组`A`和模型`model`进行矩阵乘法，得到计算的输出值`B_fit`。
    最后，代码使用`np.sum`函数计算每个数据点的误差平方和，并将结果存储在`err_per_point`数组中。该数组的维度与输入数据`data`的维度相同，其中每个元素表示对应数据点的误差平方和。
    通过返回`err_per_point`数组，可以进一步分析拟合结果，评估模型的精度和适用性。在实际应用中，误差平方和可以用于评估模型在不同数据点上的拟合效果，以及与其他模型进行比较。
    '''

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = sp.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point


def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    '''
    A_exact:  变量x 
        `random.random()` 函数来生成一个形状为 (n_samples, n_inputs) 的二维数组，其中每个元素的值都是在 0 到 1 之间均匀分布的随机浮点数。                  
                  因此，`A_exact = 20 * np.random.random((n_samples, n_inputs))` 会将生成的二维数组中的每个元素都乘以 20，得到一个新的二维数组 `A_exact`。
                  其中每个元素的值都是在 0 到 20 之间均匀分布的随机浮点数。
         np.random.random((n, m))
                  其中，n和m是整数，表示生成的随机数数组的行数和列数。这个函数返回一个由随机数组成的二维数组，其中每个随机数的取值范围在 0 到 1 之间，均匀分布。
    A_exact -> shape=(500,1) [[5.95710282e+00], [1.09638407e+01], [1.39324451e+01],...]   
    '''
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量

    '''
    perfect_fit: 斜率k 
        `random.normal()` 函数来生成一个形状为 (n_inputs, n_outputs) 的二维数组，其中 n_inputs 和 n_outputs 是输入特征和输出特征的数量。
                  因此，`perfect_fit = 60 * np.random.normal( size = (n_inputs, n_outputs) )` 会将生成的二维数组中的每个元素都乘以 60，
                  其中每个元素的值都是在 0 到 60 之间均匀分布(每个元素都是一个服从均值为 0，标准差为 1 的正态分布的)的随机浮点数。
        `np.random.normal()`用于生成一个服从正态分布的随机数数组。正态分布也称为高斯分布，是一种常见的概率分布模型，其特征是中心对称，平均值为 0，标准差为 1。
            函数的语法如下：np.random.normal(mean, std, size) :
            其中，`mean`表示平均值，`std`表示标准差，`size`表示返回的随机数数组的大小。如果`size`为一个整数，则返回一个一维数组；如果`size`为一个元组，则返回一个多维数组。
    perfect_fit -> shape=(1,1) [[8.87969873]]   
    '''
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    '''
    假设数学模型为线性回归模型，根据随机生成的变量x(A_exact)和随机生成的"均匀分布"的斜率k(perfect_fit)计算得到y的值(B_exact)
    B_exact: 变量y = k * x 
      A_exact = 500 X 1   perfect_fit = 1 X 1   =》  B_exact = 500 X 1
    B_exact -> shape=(500,1) [[5.28972783e+01], [9.73556025e+01], [1.23715915e+02], ...]
    '''
    B_exact = sp.dot(A_exact, perfect_fit)  # y = x * k

    '''
    加入高斯噪声,最小二乘能很好的处理 生成所有的点（A_noisy, B_noisy）
    A_noisy -> （500，1）[[ 6.45044495e+00], [ 1.16991225e+01], [ 1.41763188e+01]...]
    B_noisy ->  (500,1) [[ 53.36293024], [ 97.86555685], [122.98980501]...]
    '''
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        # 添加"局外点"  100个离群数据 （Xi,Yi）
        n_outliers = 100
        '''
        arange()函数的语法为np.arange(start, stop, step)，其中start表示起始值，stop表示结束值，step表示步长。如果没有指定step，则默认步长为1。
        生成一个从0到A_noisy.shape[0] - 1的整数序列，即从0到A_noisy的行数减1。
        '''
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        '''random.shuffle()函数的语法为np.random.shuffle(x)，其中x是要进行随机重排的数组。'''
        np.random.shuffle(all_idxs)  # 将all_idxs打乱   [0 1 2 ...]-> [379 406 220 ...]
        outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点  [379 406 220 ...
        '''
        `np.random.random((n_samples, n_inputs))` 会返回一个形状为 (n_samples, n_inputs) 的二维数组，其中每个元素的值都是在 0 到 1 之间均匀分布的随机浮点数。
        `np.random.normal()`用于生成一个服从正态分布的随机数数组。正态分布也称为高斯分布，是一种常见的概率分布模型，其特征是中心对称，平均值为 0，标准差为 1。
        从500个点中取100个点加入噪声（Xi，Yi）
        A_noisy[outlier_idxs] -> (500,1) [[ 6.45044495e+00], [ 1.16991225e+01], [ 1.57308118e+01]...]
        B_noisy[outlier_idxs] -> (500,1) [[  53.36293024], [  97.86555685], [ -15.45516165]...]
        '''
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi

    # setup model （Xi,Yi）
    '''
    np.hstack()是一个用于在水平方向上连接数组的函数。它将多个数组在水平方向上连接在一起，生成一个新的数组。
    生成所有点500（其中100个设置为离群点）
    all_data -> (500,2) [[ 6.45044495e+00  5.33629302e+01], [ 1.16991225e+01  9.78655569e+01], [ 1.57308118e+01 -1.54551616e+01], ...]
    '''
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  # 数组的第一列x:0    range(0,1)
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1   [1]
    debug = False

    # 最小二乘法 算法
    '''
    实例化了一个`LinearLeastSquareModel`类(自定义)，用于实现线性最小二乘回归模型的。根据代码，这个类的实例化过程需要三个参数：
        1. `input_columns`：输入特征的列数。
        2. `output_columns`：输出特征的列数。
        3. `debug`：一个布尔值，用于控制是否启用调试模式。
    model ->  <__main__.LinearLeastSquareModel object at 0x000002B9B8440040>    
    '''
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型
    '''
    `linalg.lstsq()` 函数来执行线性最小二乘拟合。这个函数可以找到一个最佳的线性模型，使得模型的预测值与目标值之间的差异最小。
        原型为sp.linalg.lstsq(a, b, rcond='warn')，其中a为系数矩阵的二维数组，b为代表因变量的一维或二维数组，rcond为在计算广义逆矩阵时使用的奇异值截断值。
                            默认情况下，rcond='warn'指的是当遇到很小的奇异值时，会产生警告信息。
                            all_data[:, input_columns]：这是一个二维数组，它包含了所有数据的输入列，指定了要使用的输入特征的列索引。
                            all_data[:, output_columns]：这也是一个二维数组，它包含了所有数据的输出列，指定了要预测的目标变量的列索引
        该函数返回一个包含多个数组的元组，其中包括解x、残差平方和、矩阵的秩以及矩阵的奇异值。
        1. `linear_fit`：这是一个返回的数组，表示最佳的线性模型的参数。 k = [[7.14628212]]
                         在这个例子中，它是一个包含 `input_columns` 个元素的数组。这些元素是模型的斜率，也就是每个输入特征对输出特征的影响程度。
        2. `resids`：这是一个返回的数组，表示模型的残差。残差是模型的预测值与目标值之间的差异。 ri = [1127668.75617866]
                         在这个例子中，它是一个包含 `n_samples` 个元素的数组，其中 `n_samples` 是数据集中的样本数量。
        3. `rank`：这是一个返回的整数，表示矩阵 `all_data[:, input_columns]` 的秩。矩阵的秩是其非零奇异值的数量。 rank = 1 
                         在这个例子中，如果 `rank` 等于 `input_columns`，那么说明矩阵是满秩的，也就是说，输入特征之间是线性无关的。
        4. `s`：这是一个返回的浮点数，表示模型的均方误差（Mean Squared Error，MSE）。MSE 是残差的平方和的平均值，它是模型预测的平均误差程度的度量。 s = [256.19721757]
    这些输出可以用于评估线性拟合的质量，并根据需要进行进一步的分析和处理。
    '''
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
    '''
    RANSAC（Random Sample Consensus，随机抽样一致性）算法，对`all_data`数据进行拟合。其中，`ransac_fit`是拟合结果，`ransac_data`是经过拟合后的数据集。
        RANSAC 算法是一种用于估计数学模型参数的迭代方法，其基本思想是：从一组包含噪声的样本数据中，通过重复抽样，选择使误差最小的样本点来估计模型参数。
        - `all_data`：待拟合的数据集。
        - `model`：拟合的模型，例如直线拟合中的`y=kx+b`。
        - `50`：每次迭代中随机选择的样本数量。
        - `1000`：最大迭代次数。
        - `7e3`：最大误差容忍度。
        - `300`：最小内点数。需要的最小内群点数（停止条件）, 满足时进入比较。
        - `debug=debug`：控制算法的输出，若为`True`则输出调试信息。
        - `return_all=True`：控制算法的返回值，若为`True`则返回所有的拟合结果，否则只返回最佳拟合结果。
    而`ransac_fit`和`ransac_data`是返回的参数，分别表示最佳拟合结果和经过拟合后的数据集。
    '''
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

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


if __name__ == "__main__":
    test()
