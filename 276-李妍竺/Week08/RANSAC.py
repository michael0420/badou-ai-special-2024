import numpy as np
import scipy as sp
import scipy.linalg as sl

#1 定义ransac
def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    #初始化迭代次数、最优拟合模型、最优误差、最优内群点索引。
    iterations =0
    bestfit = None
    besterr = np.inf #设置一个默认值。inf代表极大
    best_inlier_idxs = None

    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n,data.shape[0])   # random函数将数据分成n 和剩余

        print('maybeidx', maybe_idxs)

        maybe_inliers = data[maybe_idxs,:]  #[,:]： 列不变。 代表着选取maybe_indxs所在的行，取出这些行，也就是这些点的数据
        maybe_model = model.fit(maybe_inliers) # 使用可能的内群点拟合模型

        # 计算测试点与拟合模型的误差。
        test_points = data[test_idxs,:]  # [test_idxs,:] 等价于 [test_idxs]  其余数据点
        test_err = model.get_error(test_points,maybe_model)

        # 根据误差阈值t将测试点分类为内群点或者外群点。
        also_idxs = test_idxs[test_err<t]
        also_inliers = data[also_idxs,:]

        if debug:   #查看报错内容
            print('test_err.max()', test_err.max())
            print('mean(test_err)',np.mean(test_err))
            print('iteration',iterations)
            print('len(also_inliers)',len(also_inliers))

        print('d = ', d)  # 判定拟合良好的样本数阈值
        # 如果内群点数量满足阈值 d，则认为找到了更好的模型。
        if (len(also_inliers) > d):
            # 将可能的内群点和新发现的内群点连接起来，重新拟合模型并计算新的误差。
            betterdata = np.concatenate((maybe_inliers,also_inliers)) # 样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            newerr = np.mean(better_errs)  #平均误差作为新的误差
            # 如果新的误差比最优误差要小，则更新最优拟合模型和最优误差。
            if newerr < besterr:
                bestfit = bettermodel  #更新拟合
                besterr = newerr   #更新误差
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))   # 更新局内点,将新点加入
            iterations +=1   #迭代次数加1，进入下一次循环

    # 如果未找到最优拟合模型，则抛出异常。
    if bestfit is None:
        raise ValueError('未找到最优拟合')

    # 根据设定来，此代码中设定为False，所以返回最佳拟合。   如果为True，则还返回内群点的索引
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit

# 2定义random_partition 函数，用于将数据随机分为两部分。
# 函数首先生成一个从 0 到 n_data - 1 的索引数组，然后随机打乱这个数组。
# 接着取前 n 个索引作为第一部分，剩余的索引作为第二部分。

def random_partition(n,n_data):
    '''

    np.arange:用于生成数组： 函数返回一个有终点和起点的固定补偿的排列。
              一个参数时，参数为终点，起点默认为0，步长取默认1  ：可用于获取索引
              两个参数时，第一个为起点，第二个终点，步长取默认值1
              三个参数时，起点，终点，步长（支持小数） ：可构建等差数列
    '''
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)  #打乱元素顺序， 多维时，打乱第一维顺序
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

# 3 输入模型，此处用最小二乘

class Linearleastsquare_model:
    def __init__(self,input_x,output_y,debug = False):
        self.input_x = input_x
        self.output_y = output_y
        self.debug = debug

    def fit(self, data):    #此处的data是maybe_inliers
        # vstack:垂直方向
        A = np.vstack([data[:,i] for i in self.input_x]).T  #其实就是maybe_inliers的第一列，还是垂直放，但格式变成了8位
        B = np.vstack([data[:,i] for i in self.output_y]).T     # 第二列Yi-->行Yi

        #调用sl中的最小二乘法
        '''
        输入参数：
        a：代表设计矩阵。 X
        b：代表观测值。   Y
        cond：float，控制矩阵的奇异值截断。
        check_finite：bool or 'warn'，指示是否检查输入数组是否包含有限数值，若为 'warn'，则会发出警告。
        返回值：
        x：最小二乘解。  长度为2： 斜率，截距
        resid：残差的和的平方。
        rank：int，代表设计矩阵的秩。
        s：特征值
        '''
        x, resids, rank, s = sl.lstsq(A, B)
        return x  # 返回斜率与截距所组成的向量

    def get_error(self,data,model):      #此处的data是test_points, model是调用了fit的maybemodel
        A = np.vstack([data[:, i] for i in self.input_x]).T
        B = np.vstack([data[:, i] for i in self.output_y]).T
        B_fit = np.dot(A,model)   # 计算的y值,B_fit = model.k*A + model.b   此处的model通过ransac调用为x
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  #此处每行一个数据，sum的作用在于去括号，axis=1,去掉里层括号，数据变成了行]
        return err_per_point



def test():
    #生成数据
    n_samples = 500
    n_inputs = 1   #每个样本 输入一个X，输出一个Y
    n_outputs = 1
    A_input = 30 * np.random.random(size=(n_samples, n_inputs)) #随机生成0-30之间的500行1列数据 500个行向量  X
    Coeff = 50*np.random.normal(size=(n_inputs,n_outputs))  #用正态分布，随机一个斜率
    B_output = np.dot(A_input,Coeff)   # y = x * k

    # 加入高斯噪声,也是创建随机数的过程，可以打乱固定的k关系
    A_noisy = A_input + np.random.normal(size=A_input.shape)   # 500 * 1行向量,代表Xi
    B_noisy = B_output + np.random.normal(size=B_output.shape)  # 500 * 1行向量,代表Yi

    # if 1 等于if true:  相当于一个占位符，便于后序更改
    if 1:
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idxs)  # 将all_idxs打乱
        # 添加 异常值，一些比较离谱的数据
        n_outliers = 100
        outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 选出来的这100个点，再换个数据
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

     # Model
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_x = range(n_inputs)      #也可直接：[0]     range(1)  第一列，索引：0
    output_y = [n_inputs + i for i in range(n_outputs)]    # 数组最后一列y:1    输出值为：[1]
    debug = False
    model = Linearleastsquare_model(input_x, output_y, debug=debug)  # 类的实例化:用最小二乘生成已知模型
    linear_fit, resids, rank, s = sl.lstsq(all_data[:, input_x], all_data[:, output_y]) # 所有数据的拟合结果

    # 运行RANSAC
    ransac_fit, ransac_data = ransac(all_data, model, 50,1000,7000,300,debug=debug,return_all = True)


    if 1:
        import pylab   # numpy, scipy, matplotlib模块的合集

        sort_idxs = np.argsort(A_input[:,0]) #从小到大排列后的索引
        A_col0_sorted = A_input[sort_idxs]  # 利用索引，将输入点重新排序  从小到大

        if 1:
            pylab.plot(A_noisy[:,0],B_noisy[:,0],'k.',label = 'data')    #k:黑色 .点  ，像素
            pylab.plot(A_noisy[ransac_data['inliers'],0],B_noisy[ransac_data['inliers'],0],'rx',label='RANSAC data')
        else:   # 如果没有添加异常值   本代码目前无non_outlier
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A_col0_sorted[:,0], np.dot(A_col0_sorted,ransac_fit)[:,0],label='ransac fit')
        pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, Coeff)[:, 0], label='perfect systerm-coeff')
        pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit)[:, 0], label='linear fit')
        pylab.legend()   #添加图例
        pylab.show()


if __name__ == "__main__":
    test()