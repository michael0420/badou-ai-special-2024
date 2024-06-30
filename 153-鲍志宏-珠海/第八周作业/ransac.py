import numpy as np
import scipy as sp
import scipy.linalg as sl

#定义RANSAC算法的主函数
def ransac(data, model, n, k, t, d, debug = False, return_all = False):

    #初始化迭代次数、最佳拟合模型、最小误差和最佳内点索引
    iterations = 0
    bestfit = None
    # 初始化最小误差为正无穷大
    besterr = np.inf
    best_inlier_idxs = None

    #开始迭代
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])#随机选择n个数据点用于模型拟合，其余数据点用于验证
        print ('test_idxs = ', test_idxs)
        maybe_inliers = data[maybe_idxs, :] #提取用于拟合的数据点
        test_points = data[test_idxs] #提取用于验证的数据点
        maybemodel = model.fit(maybe_inliers) #用拟合数据点训练模型
        test_err = model.get_error(test_points, maybemodel) #计算误差:平方和最小
        print('test_err = ', test_err <t)
        also_idxs = test_idxs[test_err < t]#找出误差小于阈值 t 的验证数据点的索引，称为内点
        print ('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs,:]#提取内点数据
        if debug:
            print ('test_err.min()',test_err.min())
            print ('test_err.max()',test_err.max())
            print ('numpy.mean(test_err)',numpy.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d' %(iterations, len(also_inliers)) )
        print('d = ', d)

        #如果内点数量超过阈值d，用所有内点重新拟合模型并计算平均误差
        if (len(also_inliers) > d):
            betterdata = np.concatenate( (maybe_inliers, also_inliers) ) #样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs) #平均误差作为新的误差
            #如果新的平均误差比当前最小误差更小，则更新最佳模型和内点
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate( (maybe_idxs, also_idxs) ) #更新局内点,将新点加入
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit,{'inliers':best_inlier_idxs}
    else:
        return bestfit
 
#将数据随机划分为两个部分，返回两个索引数组
def random_partition(n, n_data):
    all_idxs = np.arange(n_data) #生成数据索引
    np.random.shuffle(all_idxs) #打乱索引顺序
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

#定义一个用于线性最小二乘拟合的模型类，初始化输入和输出列索引
class LinearLeastSquareModel:
    #最小二乘求线性解,用于RANSAC的输入模型    
    def __init__(self, input_columns, output_columns, debug = False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    #使用最小二乘法拟合模型并返回拟合参数
    def fit(self, data):
		#np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T #构建输入数据矩阵
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T #构建输出数据矩阵
        x, resids, rank, s = sl.lstsq(A, B) #进行最小二乘拟合
        return x #返回最小平方和向量   
    #计算数据点与模型拟合结果之间的误差
    def get_error(self, data, model):
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T
        B_fit = sp.dot(A, model) #计算拟合结果
        err_per_point = np.sum( (B - B_fit) ** 2, axis = 1 ) #计算每个点的误差
        return err_per_point

#测试函数
def test():
    #生成理想数据
    n_samples = 500 #样本个数
    n_inputs = 1 #输入变量个数
    n_outputs = 1 #输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))#随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal( size = (n_inputs, n_outputs) ) #随机线性度，即随机生成一个斜率
    B_exact = sp.dot(A_exact, perfect_fit) # y = x * k
 
    #加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal( size = A_exact.shape ) #500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal( size = B_exact.shape ) #500 * 1行向量,代表Yi
 
    if 1:
        #添加"局外点"
        n_outliers = 100
        all_idxs = np.arange( A_noisy.shape[0] ) #获取索引0-499
        np.random.shuffle(all_idxs) #将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers] #100个0-500的随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random( (n_outliers, n_inputs) ) #加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal( size = (n_outliers, n_outputs)) #加入噪声和局外点的Yi
    #setup model 
    all_data = np.hstack( (A_noisy, B_noisy) ) #形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  #数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)] #数组最后一列y:1
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug = debug) #类的实例化:用最小二乘生成已知模型
 
    linear_fit,resids,rank,s = sp.linalg.lstsq(all_data[:,input_columns], all_data[:,output_columns])
    
    #run RANSAC 算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug = debug, return_all = True)
 
    if 1:
        import pylab
 
        sort_idxs = np.argsort(A_exact[:,0])
        A_col0_sorted = A_exact[sort_idxs] #秩为2的数组
 
        if 1:
            pylab.plot( A_noisy[:,0], B_noisy[:,0], 'k.', label = 'data' ) #散点图
            pylab.plot( A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label = "RANSAC data" )
        else:
            pylab.plot( A_noisy[non_outlier_idxs,0], B_noisy[non_outlier_idxs,0], 'k.', label='noisy data' )
            pylab.plot( A_noisy[outlier_idxs,0], B_noisy[outlier_idxs,0], 'r.', label='outlier data' )
 
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,ransac_fit)[:,0],
                    label='RANSAC fit' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,perfect_fit)[:,0],
                    label='exact system' )
        pylab.plot( A_col0_sorted[:,0],
                    np.dot(A_col0_sorted,linear_fit)[:,0],
                    label='linear fit' )
        pylab.legend()
        pylab.show()
 
if __name__ == "__main__":
    test()
