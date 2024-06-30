import numpy as np
import cv2
import scipy.linalg as sl
import scipy as sp


# def random_partition(n, n_data):
#     # n_data是一个整数
#     # 生成一个[0,n_data)的数组
#     all_idxs = np.arange(n_data)
#     # 打乱这个数组中元素的顺序
#     np.random.shuffle(all_idxs)
#     # 取前n个构成一个一维数组idxs1
#     idxs1 = all_idxs[:n]
#     # 取n后面的元素构成一个一维数组idxs2
#     idxs2 = all_idxs[n:]
#     return idxs1, idxs2
#
#
# class LinearLeastSquareModel():
#     # 用最小二乘法求线性解，用于ransac的输入模型
#     def __init__(self, input_columns_index, output_columns_index, debug):
#         self.input_columns_index = input_columns_index
#         self.output_columns_index = output_columns_index
#         self.debug = debug
#
#     def fit(self, data):
#         # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
#         # A，B是一个列向量，因为首先我们取出来的是data的列
#         # 例如：
#         # [1]
#         # [2]
#         # [3]
#         # 但是np.vstack会把它变成行向量，即横着放置，即变成[1,2,3]
#         # 最后再转置得到:
#         # [1]
#         # [2]
#         # [3]
#         # B同理
#         # 第一列，因为input_columns_index是一个数组，但是只有一个元素[0],代表对于该函数模型只有一个输入
#         # A = np.vstack([data[:, i] for i in self.input_columns_index]).T
#         A=np.vstack([data[:, i] for i in self.input_columns_index]).T
#         # 第二列，因为output_columns_index是一个数组，但是只有一个元素[1]，代表对于该函数模型只有一个输出
#         # B = np.vstack([data[:, i] for i in self.output_columns_index]).T
#         B=np.vstack([data[:, i] for i in self.output_columns_index]).T
#
#         # x是一个列表
#         # 调用最小二乘法的接口拟合一个模型，例如y=kx模型
#         # x：线性回归的系数，也就是解向量 x，它使得 ||Ax - b|| 最小，其中 A 是输入数据矩阵，b 是目标数据矩阵。
#         # residues：残差平方和，即 ||Ax - b||^2，衡量了模型拟合的好坏。
#         # rank：输入数据矩阵 A 的秩。
#         # s：A 的奇异值。
#         # 即输入一堆[x,y]的点，拟合出一个近似的函数模型即为最小二分法
#         x, residues, rank, s = sl.lstsq(A, B)
#         return x
#
#     def get_error(self, data, model):
#         # np.vstack([data[:, 1])会拿出第一列，但是这一列会按照行排布
#         # 例如第一列为
#         # [1]
#         # [2]
#         # [3]
#         # 但是拿出来之后会把它放平，即变成[1,2,3]的一维数组，所以需要再把它转置成列向量
#         # [1]
#         # [2]
#         # [3]
#         # 第一列，因为input_columns_index是一个数组，但是只有一个元素[0],代表对于该函数模型只有一个输入
#         A = np.vstack([data[:, i] for i in self.input_columns_index]).T  # 第一列Xi
#         # 第二列，因为output_columns_index是一个数组，但是只有一个元素[1]，代表对于该函数模型只有一个输出
#         B = np.vstack([data[:, i] for i in self.output_columns_index]).T  # 第二列Yi
#         B_fit = np.dot(A, model)  # 得到一个数组，每一个元素表示x输入这个model之后得到的y的值
#
#         # axis=1表示沿着列求和，也就是按行求和的意思，这里容易混淆
#         err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # 计算残差平方和，返回值是一个一维数组
#
#         # print("***********************")
#         # print(f'A.shape={A.shape}')
#         # print("***********************")
#         # print("***********************")
#         # print(f'B.shape={B.shape}')
#         # print("***********************")
#         # print("***********************")
#         # print(f'B_fit.shape={B_fit.shape}')
#         # print("***********************")
#         # print("***********************")
#         # print(f'err_per_point.shape={err_per_point.shape}')
#         # print("***********************")
#         return err_per_point  # 以下是解释
#
#         # 解析：A 和 B 都是形状为 (450, 1) 的二维数组（也可以看作是每行一个元素的列向量）。
#         # 当您使用 np.sum((B - B_fit) ** 2, axis=1) 计算残差平方和时，axis=1 表示沿着第
#         # 二个轴（即列）进行求和。
#         # 由于 B - B_fit 的结果也是一个形状为 (450, 1) 的二维数组（因为两个操作数都是
#         # (450, 1) 形状），沿着第二个轴（即列）进行求和实际上就是对每一行进行求和。因为每行
#         # 都只有一个元素，所以求和的结果就是一个一维数组，其中包含了450个元素，每个元素都是对
#         # 应行中唯一元素的平方残差。
#         # 换句话说，np.sum((B - B_fit) ** 2, axis=1) 实际上是在计算每个点的残差平方和，但由
#         # 于每个点只对应一个值（因为 B 和 B_fit 都是列向量），所以结果就是一个一维数组。
#         # 如果您想要得到一个二维数组，可能不需要使用 axis=1 进行求和，但通常在这种情况下，我们
#         # 确实想要得到一个表示每个点残差平方和的一维数组。
#         # 总结起来，err_per_point 是一个一维数组的原因是沿着 B - B_fit 的列（即第二个轴）进行
#         # 了求和，而每一列都只有一个元素。
#
#
# # 输入:
# #     data - 样本点
# #     model - 假设模型:事先自己确定
# #     n - 生成模型所需的最少样本点
# #     k - 最大迭代次数
# #     t - 阈值:作为判断点满足模型的条件
# #     d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
# def ransac(data, model, n, k, t, d, debug=False, return_all=False):
#     # 记录已经迭代了iterations次
#     iterations = 0
#     # 记录最好(匹配)的模型
#     bestfit = None
#     # 记录最小的误差
#     besterr = np.inf
#     # 用来记录内点
#     best_inlier_idxs = None
#     # 循环迭代k次，因为我们不可能永远迭代下去，时间成本，硬件资源成本
#     while iterations < k:
#         # 随机生成一个下标数组，选取其中的一些随机点，用于生成一个待拟合的函数，
#         # 因为有一些点之后就可以暂时推导出一个函数了，然后再用其它的点带入该函数
#         # 看看其它的点对于这个函数的拟合效果如何，如果有很多的点的x代入该函数，
#         # 得到的y值的误差都比较小，说明拟合效果很好，那么这个函数图像的大致趋势就
#         # 很可能符合这些点
#         maybe_idxs, test_idxs = random_partition(n, data.shape[0])
#         # print('test_idxs = ', test_idxs)
#         # 根据返回的随机下标，取出对应的点，maybe_inliers中的每一个元素都是对应的一个[Xi,Yi]点，
#         # 对于现在我们的假设而言，这些点我们假设它为内点
#         maybe_inliers = data[maybe_idxs, :]
#         # test_points中的每一个元素也是对应test_idxs下标中的一个[Xi,Yi]点，用于后面的测试
#         test_points = data[test_idxs, :]
#         # 用maybe_inliers中的[Xi,Yi]点，拟合出一个假如是y=kx+b的模型，所以返回值maybemodel
#         # 是已知k和b参数的了
#         maybemodel = model.fit(maybe_inliers)
#         # 我们用一些测试点[Xi,Yi]测试该模型，求出残差平方和（也就是误差）,这是一个一维数组[err1,err2...]
#         test_err = model.get_error(test_points, maybemodel)
#         # print("------------------------------------")
#         # print(test_err.shape)
#         # print("------------------------------------")
#         # print('test_err = ', test_err < t)
#         # also_idxs数组包含了 test_idxs 中所有满足 test_err < t 条件的原始索引值
#         # 把所有的误差小于t阈值的下标全部找出来，also_idxs是一个只有一列的列向量
#         also_idxs = test_idxs[test_err < t]
#         # print('also_idxs = ', also_idxs)
#         # 根据下标取出对应行的值[X,Y]，代表测试集有多少个符合该函数的点，即内点
#         also_inliers = data[also_idxs, :]
#         if debug:
#             print('test_err.min()', test_err.min())
#             print('test_err.max()', test_err.max())
#             print('numpy.mean(test_err)', np.mean(test_err))
#             print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
#         # 如果内点的数目比阈值d大，即符合这个函数的内点数目大于最低的阈值要求
#         if len(also_inliers) > d:
#             # 把maybe_inliers和also_inliers中的所有点拼接成一个数组
#             betterdata = np.concatenate((maybe_inliers, also_inliers))
#             # 用所有的内点再拟合出一个更贴近这些点的函数模型，例如求出更符合y=kx+b中的参数k和b
#             bettermodel = model.fit(betterdata)
#             # 求出各个点相对于这个模型的误差，返回的better_errs是一个误差数组
#             better_errs = model.get_error(betterdata, bettermodel)
#             # 求出误差数组的平均值
#             thiserr = np.mean(better_errs)
#             # 用平均值和之前定义的最小的误差进行比较，如果小于，就更新最小误差，最好模型，最符合函数模型的内点下标
#             if thiserr < besterr:
#                 bestfit = bettermodel
#                 besterr = thiserr
#                 best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
#         # 迭代次数+1
#         iterations += 1
#     # 没有找到一个很好的拟合模型，那就抛出一个异常
#     if bestfit is None:
#         raise ValueError("did't meet fit acceptance criteria")
#     # 这个return_all是传进来的参数
#     if return_all:
#         return bestfit, {'inliers': best_inlier_idxs}
#     else:
#         return bestfit
#
#
# # data=[[1,2],[3,4],[5,6]]
# # ransac(data,)
#
#
# def test():
#     # 生成理想数据
#     n_sample = 500  # 样本个数
#     n_input = 1  # 输入变量个数
#     n_output = 1  # 输出变量个数
#     # A_exact = 20 * np.random.random((n_sample, n_input))
#     A_exact = 20 * np.random.random((n_sample, n_input))  # 随机生成0-20之间的500*1个数据:500*1的列向量
#     # 这个是正确答案的斜率，用来后面画图
#     perfect_fit = 60 * np.random.normal(size=(n_input,n_output))  # 随机线性度，即随机生成一个斜率，即随机生成一个数
#     # 求出这个函数，这个是正确答案，因为我们的数据一开始就是完全符合这条直线的
#     B_exact = np.dot(A_exact, perfect_fit)
#     # 加入高斯噪声，即不要让所有的点都刚好在这条直线上，让它们有点误差
#     # 生成和A_exact.shape格式一样的多个随机数，并且加到对应的A_exact上
#     A_noise = A_exact + np.random.normal(size=A_exact.shape)
#     B_noise = B_exact + np.random.normal(size=B_exact.shape)
#
#     if True:
#         # 添加局外点
#         n_outliers = 100
#         all_index = np.arange(0, A_noise.shape[0])  # 获取索引0-499
#         np.random.shuffle(all_index)  # 将all_idxs打乱
#         outliers_index = all_index[:n_outliers]  # 100个0-499的随机局外点
#         # 随机选n_outliers个位置，把它们函数值变得差异大一点，变成局外点
#         A_noise[outliers_index] = 20 * np.random.random(size=(n_outliers, n_input))
#         B_noise[outliers_index] = 50 * np.random.normal(size=(n_outliers, n_output))
#
#     # 建立模型
#     # np.hstack((A_noise, B_noise)) 表示把A_noise和B_noise水平放置
#     # 假如：A_noise = np.array([[1, 2], [3, 4]])
#     #      B_noise = np.array([[5], [6]])
#     #      得到的all_data就是：
#     #      [1,2,5]
#     #      [3,4,6]
#     # 所以all_data的结构：
#     # A_noise    B_noise
#     # 125        457
#     # 453        123
#     # 123        325
#     # 489        234
#     # ...
#     all_data = np.hstack((A_noise, B_noise))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
#     # input_columns_index是一个一维数组，[0,1,2...n-1]
#     # 数组的第一列下标,为什么？其实不一定是数组的最后一列，只有当n_input=1时才表示第一列，n_input
#     # 等于n时，代表前n列的下标
#     input_columns_index = range(n_input)
#     # output_columns_index是一个一维数组，[...,n-3,n-2,n-1]
#     # 数组最后一列下标，只有当n_output=1时才表示最后一列，当n_output=n时代表最后n列的下标
#     output_columns_index = [n_input + i for i in range(n_output)]
#     # print(f'input_columns_index={input_columns_index}')
#     # print(f'output_columns_index={output_columns_index}')
#     debug = False
#     # 类的实例化:用最小二乘生成已知模型
#     model = LinearLeastSquareModel(input_columns_index, output_columns_index, debug)
#
#     # 调用最小二乘法的接口拟合一个模型，例如y=kx模型
#     # linear_fit：线性回归的系数，也就是解向量 x，它使得 ||Ax - b|| 最小，其中 A 是输入数据矩阵，b 是目标数据矩阵。
#     # resids：残差平方和，即 ||Ax - b||^2，衡量了模型拟合的好坏。
#     # rank：输入数据矩阵 A 的秩。
#     # s：A 的奇异值。
#     # all_data[:, input_columns_index]：数组的第一列，即输入值，相当于x
#     # all_data[:, output_columns_index]：数组的最后一列，即输出值，相当于x对应的y
#     # 即输入一堆[x,y]的点，拟合出一个近似的函数模型即为最小二分法
#     linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns_index], all_data[:, output_columns_index])
#
#     # run ransac 算法
#     ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)
#
#     if True:
#         import pylab
#         sort_idxs = np.argsort(A_exact[:, 0])
#         A_col0_sorted = A_exact[sort_idxs]
#
#         if True:
#             pylab.plot(A_noise[:, 0], B_noise[:, 0], 'k.', label='data')
#             pylab.plot(A_noise[ransac_data['inliers'], 0], B_noise[ransac_data['inliers'], 0], 'bx',
#                        label='RANSAC data')
#         else:
#             pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
#             pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')
#
#         pylab.plot(A_col0_sorted[:, 0],
#                    np.dot(A_col0_sorted, ransac_fit)[:, 0],
#                    label='RANSAC fit')
#         pylab.plot(A_col0_sorted[:, 0],
#                    np.dot(A_col0_sorted, perfect_fit)[:, 0],
#                    label='exact system')
#         pylab.plot(A_col0_sorted[:, 0],
#                    np.dot(A_col0_sorted, linear_fit)[:, 0],
#                    label='linear fit')
#         pylab.legend()
#         pylab.show()
#

# if __name__ == '__main__':
#     test()


# # 第二次
# import scipy as sp
# import scipy.linalg as sl
# import pylab
#
# class LinearLeastSquareModel:
#     def __init__(self,input_col_idx,output_col_idx,debug=False):
#         self.input_col_idx=input_col_idx
#         self.output_col_idx=output_col_idx
#         self.debug=debug
#     def fit(self,A_B_exact):
#         # x包含了最小二乘法的解，可以认为包含了y = kx + b中的k和b的值，x[0] = k, x[1] = b
#         A_exact=np.vstack(A_B_exact[:,i] for i in self.input_col_idx).T
#         B_exact=np.vstack(A_B_exact[:,i] for i in self.output_col_idx).T
#         x,_,_,_=sl.lstsq(A_exact,B_exact)
#         return x
#         pass
#     # model包含了最小二乘法的解，可以认为包含了y=kx+b中的k和b的值，model[0]=k,model[1]=b,
#     # 是经过fit函数计算出来的
#     def get_error(self,A_B_exact,model):
#         A_exact=np.vstack(A_B_exact[:,i] for i in self.input_col_idx).T
#         B_exact=np.vstack(A_B_exact[:,i] for i in self.output_col_idx).T
#         B_fit=np.dot(A_exact,model)
#         error=np.sum((B_fit-B_exact)**2,axis=1)
#         return error
#         pass
#
# # bug，这里的参数曾经传翻了，哭死
# def random_partition(min_sample_num,n):
#     idx=np.arange(0,n)
#     np.random.shuffle(idx)
#     maybe_idx=idx[:min_sample_num]
#     test_idx=idx[min_sample_num:]
#     return maybe_idx,test_idx
#
# def ransac(data, model, min_sample_num, max_iterator, min_threshold, d, debug = False, return_all = False):
#     iterators=0
#     best_fit=None
#     best_err=np.inf
#     best_inlier_idx=None
#     while iterators<max_iterator:
#         # 随机选取min_sample_num个作为内点
#         maybe_idx,test_idx=random_partition(min_sample_num,data.shape[0])
#         maybe_inlier_data=data[maybe_idx,:]
#         test_points_data=data[test_idx,:] #?
#         maybe_model=model.fit(maybe_inlier_data)
#         test_err=model.get_error(test_points_data,maybe_model)
#         also_idx=test_idx[test_err<min_threshold]
#         also_inliers_data=data[also_idx,:]
#         if len(also_inliers_data)>d:
#             better_data=np.concatenate((maybe_inlier_data,also_inliers_data))
#             better_model=model.fit(better_data)
#             better_err=model.get_error(better_data,better_model)
#             this_err=np.mean(better_err)
#             if this_err<best_err:
#                 best_fit=better_model
#                 best_err=this_err
#                 best_inlier_idx=np.concatenate((maybe_idx,also_idx))
#         iterators+=1
#
#     if best_fit is None:
#         raise ValueError("did't meet fit acceptance criteria")
#     if return_all:
#         return best_fit,{"inliers":best_inlier_idx}
#     else:
#         return best_fit
#
# if __name__=='__main__':
#     # 生成500个点数据
#     n_data=500
#     # 输入特征数
#     n_input=1
#     # 输出特征数
#     n_output=1
#     A_exact=20*np.random.random((n_data,n_input))
#     k=60*np.random.normal(size=(n_input,n_output))
#     B_exact=sp.dot(A_exact,k)
#
#     # 生成一些误差
#     A_exact_noise=A_exact+np.random.normal(size=A_exact.shape)
#     B_exact_noise=B_exact+np.random.normal(size=B_exact.shape)
#
#     # 生成100个局外点
#     n_outlier=100
#     all_idx=np.arange(0,n_data)
#     np.random.shuffle(all_idx)
#     outlier_idx=all_idx[:n_outlier]
#     A_exact_noise[outlier_idx]=20*np.random.random((n_outlier,n_input))
#     B_exact_noise[outlier_idx]=50*np.random.normal(size=(n_outlier,n_output))
#     all_data=np.hstack((A_exact_noise,B_exact_noise))
#
#     input_col_idx=range(n_input)
#     output_col_idx=[n_input+i for i in range(n_output)]
#
#     # 最小二乘法求拟合直线
#     linear_fit,_,_,_=sl.lstsq(A_exact_noise,B_exact_noise)
#     model=LinearLeastSquareModel(input_col_idx,output_col_idx)
#
#     # ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug = False, return_all = True)
#     ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug = False, return_all = True)
#
#
#     # pylab.plot(A_exact_noise[:, 0], B_exact_noise[:, 0], 'k.', label='data')
#     # sort_idxs = np.argsort(A_exact[:, 0])
#     # A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组
#     # pylab.plot(A_col0_sorted[:, 0],
#     #            np.dot(A_col0_sorted, linear_fit)[:, 0],
#     #            label='linear fit')
#     # pylab.show()
#
#     if 1:
#         import pylab
#
#         sort_idxs = np.argsort(A_exact[:, 0])
#         A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组
#
#         if 1:
#             pylab.plot(A_exact_noise[:, 0], B_exact_noise[:, 0], 'k.', label='data')  # 散点图
#             pylab.plot(A_exact_noise[ransac_data['inliers'], 0], B_exact_noise[ransac_data['inliers'], 0], 'bx',
#                        label="RANSAC data")
#         else:
#             pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
#             pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')
#
#         pylab.plot(A_col0_sorted[:, 0],
#                    np.dot(A_col0_sorted, ransac_fit)[:, 0],
#                    label='RANSAC fit')
#         pylab.plot(A_col0_sorted[:, 0],
#                    np.dot(A_col0_sorted, k)[:, 0],
#                    label='exact system')
#         pylab.plot(A_col0_sorted[:, 0],
#                    np.dot(A_col0_sorted, linear_fit)[:, 0],
#                    label='linear fit')
#         pylab.legend()
#         pylab.show()

import pylab

# 第三次

class LinearLeastSquareModel:
    def __init__(self,input_idx,output_idx,debug=False):
        self.input_idx=input_idx
        self.output_idx=output_idx
        self.debug=debug
    def fit(self,data):
        # 把数据分离出输入数据和对应的输出数据(标签)
        A_exact=np.vstack([data[:,i] for i in self.input_idx]).T
        B_exact=np.vstack([data[:,i] for i in self.output_idx]).T
        x,_,_,_=sl.lstsq(A_exact,B_exact)
        return x
        pass
    def get_error(self,data,model):
        A_exact=np.vstack([data[:,i] for i in self.input_idx]).T
        B_exact=np.vstack([data[:,i] for i in self.output_idx]).T
        B_fit=np.dot(A_exact,model)
        error=np.sum((B_fit-B_exact)**2,axis=1)
        return error
        pass

def random_partition(n,data_range):
    index_all=np.arange(0,data_range)
    np.random.shuffle(index_all)
    maybe_idx=index_all[:n]
    test_idx=index_all[n:]
    return maybe_idx,test_idx


def ransac(data, model, n, k, t, d, debug = False, return_all = False):
    best_fit=None
    best_err=np.inf
    best_data_idx=None
    iterators=0
    while iterators<k:
        # 随机取出n个点作为内点
        maybe_idx,test_idx=random_partition(n,data.shape[0])
        maybe_data_points=data[maybe_idx,:]
        test_data_points=data[test_idx,:]
        # 用可能的内点通过最小二乘法拟合出一条直线
        maybe_fit=model.fit(maybe_data_points)
        # all_data=np.concatenate(maybe_data_points,test_data_points)
        # 算出误差
        maybe_err=model.get_error(test_data_points,maybe_fit)
        also_idx=[maybe_err<t]
        also_data=test_data_points[also_idx]
        if len(also_data)>d:
            all_data = np.concatenate((maybe_data_points, test_data_points))
            err=np.mean(model.get_error(all_data,maybe_fit))
            if err<best_err:
                best_fit=maybe_fit
                best_err=err
                best_data_idx=np.concatenate((maybe_idx,test_idx))
        iterators+=1

    if best_fit is None:
        raise ValueError("找不到拟合的直线")
    if return_all==True:
        print(best_fit)
        print(best_data_idx.shape)
        print(best_err)
        return best_fit,{'inliers':best_data_idx}
    else:
        return best_fit


def test():
    # 创建500个随机点
    n_data_total=500
    n_input=1
    n_output=1
    A_exact=20*np.random.random(size=(n_data_total,n_input))
    k=60*np.random.normal(size=(1,1))
    B_exact=np.dot(A_exact,k)
    # 加点偏差
    A_exact_noise=A_exact+np.random.normal(size=(A_exact.shape))
    B_exact_noise=B_exact+np.random.normal(size=(B_exact.shape))
    # 随机选取100个变为局外点，也就是让它们的误差变得很大
    n_outliers=100
    # 生成下标
    idx=np.arange(0,n_data_total)
    # 打乱下标
    np.random.shuffle(idx)
    # 选取前100个
    outliers_idx=idx[:n_outliers]
    # 对这100个下标对应的点的值进行重新赋值
    A_exact_noise[outliers_idx]=20*np.random.random(size=(n_outliers,n_input))
    B_exact_noise[outliers_idx]=60*np.random.normal(size=(n_outliers,n_input))
    # 使用最小二乘法拟合一条直线
    linear_fit,_,_,_=sl.lstsq(A_exact_noise,B_exact_noise)
    # 使用ransac算法拟合一条直线
    input_idx=range(n_input)
    output_idx=[n_input+i for i in range(n_output)]
    model=LinearLeastSquareModel(input_idx,output_idx)
    data=np.hstack((A_exact_noise,B_exact_noise))

    ransac_fit,ransac_data=ransac(data,model,50,1000,7e3,300,debug=False,return_all=True)
    # 画图
    pylab.plot(A_exact_noise[:, 0], B_exact_noise[:, 0], 'k.', label='data')
    pylab.plot(A_exact_noise[ransac_data['inliers'], 0], B_exact_noise[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, linear_fit)[:, 0],
               label='linear fit')
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, k)[:, 0],
               label='exact system')
    pylab.plot(A_col0_sorted[:, 0],
              np.dot(A_col0_sorted, ransac_fit)[:, 0],
              label='RANSAC fit')
    pylab.legend()
    pylab.show()
    pass

if __name__=='__main__':
    test()
