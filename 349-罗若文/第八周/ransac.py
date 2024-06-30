import numpy
import numpy as np
import scipy as sp
import scipy.linalg as sl





# 生成数据
def initData(input_count,output_count):
    #样本个数
    data_count = 500
    x_column = 20 * np.random.random((data_count,input_count)) #生成x列数据
    k = 60*np.random.normal(size=(input_count,output_count)) #生成斜率k
    y_column = np.dot(x_column,k) # y=kx 得到y列

    # 在y=kx上的点 加入噪声
    x_column_noisy = x_column + np.random.normal(size=x_column.shape)
    y_column_noisy = y_column + np.random.normal(size=y_column.shape)

    #局外点的个数
    out_count = 100
    # 获取x列的索引并且打乱
    all_index = np.arange(x_column_noisy.shape[0])
    np.random.shuffle(all_index)
    # 取前一百个作为局外点的索引
    out_index = all_index[:out_count]
    # 将500个数据中随机100加上随机数变为局外点
    x_column_noisy[out_index] = 20 * np.random.random((out_count,input_count))
    y_column_noisy[out_index] = 50 * np.random.normal(size =(out_count,output_count))
    # 将x列y列合并为二位数组
    all_data = np.hstack((x_column_noisy,y_column_noisy))
    return all_data,x_column_noisy,y_column_noisy,x_column,y_column,k

class LinearLeastSquareModel:
    def __init__(self,input_columns, output_columns):
        self.input_columns = input_columns
        self.output_columns = output_columns
    def fit(self,data):
        A = np.vstack([data[:,i] for i in self.input_columns]).T
        B = np.vstack([data[:,i] for i in self.output_columns]).T
        # 计算残差
        x,resids,ran,s = sl.lstsq(A,B)
        # 返回最小平方和向量
        return x
    def get_error(self,data,model):
        A = np.vstack([data[:,i] for i in self.input_columns]).T
        B = np.vstack([data[:,i] for i in self.output_columns]).T
        B_fit = np.dot(A,model)
        err_per_point = np.sum((B-B_fit)**2,axis=1)
        return  err_per_point

## data:计算数据 ，model:假设模型 model_sample_count:生成模拟模型采集的样本个数 max_iteration_count:最大迭代次数
# model_err_threshold: 满足模型最大的误差值 fitting_model_min_cont:拟合较好时最少的样本数
def ransac(data, model, model_sample_count, max_iteration_count, model_err_threshold,fitting_model_min_cont):
    #当前迭代次数
    iterations = 0
    #最好的模型
    bestfit = None
    #最好的误差
    besterr = np.inf #设置默认值
    # 最好数据的内群索引
    best_inlier_idxs = None
    while iterations < max_iteration_count:
        # 随机获取拟定的模型索引 以及其他计算数据的索引
        maybe_idxs, test_idxs = random_partition(model_sample_count, data.shape[0])
        # 获取拟定的模型数据 以及其他计算数据点
        maybe_inliers = data[maybe_idxs,:]
        test_points = data[test_idxs]
        # 拟定的内群数据计算得到拟定模型
        maybe_model = model.fit(maybe_inliers)
        # 计算误差值
        test_err = model.get_error(test_points,maybe_model)
        # 获取在符合误差阈值的索引 得到符合的内群数据
        also_idxs = test_idxs[test_err < model_err_threshold]
        also_inliers = data[also_idxs,:]
        print('test_err.min()', test_err.min())
        #判断是否满足要求的样本个数
        if(len(also_inliers) > fitting_model_min_cont):
            betterdata = np.concatenate((maybe_inliers,also_inliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入
        iterations += 1
        print(iterations)
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    else:
        return bestfit, {'inliers': best_inlier_idxs}



def random_partition(n, n_data):
    all_idxs = np.arange(n_data) #获取n_data下标索引
    np.random.shuffle(all_idxs) #打乱下标索引
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


if __name__ == "__main__":
    #输入变量列数，输出变量列数
    input_count,output_count = 1,1
    # 初始化数据
    all_data,A_noisy,B_noisy,A_exact,B_exact,k = initData(input_count,output_count)
    #输入列 即第一列
    input_columns = range(input_count)
    #输出列 即第一列后的列数
    output_columns = [input_count + i for i in range(output_count)]
    #生成最小二乘法模型
    model = LinearLeastSquareModel(input_columns, output_columns)

    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # 根据ransac逻辑进行计算
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300)

    import pylab

    #画图
    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组
    pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
    pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, ransac_fit)[:, 0],
               label='RANSAC fit')
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, k)[:, 0],
               label='exact system')
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, linear_fit)[:, 0],
               label='linear fit')
    pylab.legend()
    pylab.show()