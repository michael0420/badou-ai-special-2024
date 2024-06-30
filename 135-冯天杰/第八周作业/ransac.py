import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt


# 最小二乘法
class LinearLeastSquareModel:
    def __int__(self):
        pass
    def fit(self, data):
        x = np.vstack(data[:, 0])
        # print(x)
        y = np.vstack(data[:, 1])
        k, resids, rank, s = sl.lstsq(x, y)
        # print(k)
        return k

    def get_error(self, data, k):
        x = np.vstack(data[:, 0])
        y = np.vstack(data[:, 1])
        hr = np.dot(x, k)
        error = np.sum((hr-y)**2, axis=1)
        # print(error.shape)
        return error

# ransac
def ransac(data,LinearLeastSquareModel_model, n_choice, yuzhi, max_times,wise_in):
    '''
    data: 数据点
    n_choice: 内群点个数
    yuzhi :误差阈值
    max_times： 最大迭代次数
    '''

    # 初始化
    n = 0  # 迭代次数
    best_k = None
    better_error = np.inf
    best_in_data = None

    while n < max_times:
        all_index = np.arange(data.shape[0])
        np.random.shuffle(all_index)
        in_index = all_index[:n_choice]
        out_index = all_index[n_choice:]
        in_data = data[in_index, :]
        out_data = data[out_index, :]
        maybe_k = LinearLeastSquareModel_model.fit(in_data)
        # print('aaa %d' % maybe_k.shape)
        maybe_error = LinearLeastSquareModel_model.get_error(out_data, maybe_k)
        # print(maybe_error)
        also_in_data_index = out_index[maybe_error < yuzhi]
        print(also_in_data_index.shape)
        # also_in_data = out_data[also_in_data_index, :]
        also_in_data = data[also_in_data_index, :]

        if (len(also_in_data) > wise_in):
            all_in_data = np.concatenate((in_data, also_in_data))
            # all_in_data = data[all_in_index, :]
            now_k = LinearLeastSquareModel_model.fit(all_in_data)
            now_error = LinearLeastSquareModel_model.get_error(all_in_data, now_k)
            now__error = np.mean(now_error)

            if now__error < better_error:
                better_error = now__error
                best_k = now_k
                best_in_data = all_in_data
        n += 1

    if best_k is None:
        raise ValueError("did't meet fit acceptance criteria")
    else:
        return best_k, all_in_data

# 建立数据集
n = 500

x = 20 * np.random.random((n, 1))
# print(x)
k = 60 * np.random.normal(1)
y = np.dot(x, k)
# print(y)

x_gaussion = x + np.random.normal(size=x.shape)
# print(x_gaussion)
y_gaussion = y + np.random.normal(size=y.shape)
# print(type(x_gaussion))
# print(x_gaussion.shape)
n_noise = 100
xandy_index = np.arange(x_gaussion.shape[0])
# print(xandy_index)
np.random.shuffle(xandy_index)
xandy_noiseindex = xandy_index[:n_noise]
# print(xandy_noiseindex.shape)
x_gaussion[xandy_noiseindex] = x_gaussion[xandy_noiseindex] + np.random.normal(size=(n_noise, 1))
y_gaussion[xandy_noiseindex] = x_gaussion[xandy_noiseindex] + np.random.normal(size=(n_noise, 1))

# x_gaussion = np.reshape(x_gaussion,(len(x_gaussion),1))
# y_gaussion = np.reshape(y_gaussion,(len(x_gaussion),1))
data = np.hstack((x_gaussion, y_gaussion))
print(data)

LinearLeastSquareModel_model = LinearLeastSquareModel()
LinearLeastSquareModel_k = LinearLeastSquareModel_model.fit(data)
LinearLeastSquareModel_error = LinearLeastSquareModel_model.get_error(data,LinearLeastSquareModel_k)
best_k, best_in_data = ransac(data, LinearLeastSquareModel_model, 50, 7e3, 1000, 100)
# print(LinearLeastSquareModel_k)

# 绘图
plt.figure()
plt.scatter(data[:,0], data[:, 1], label= 'data')
plt.plot(x, y, c="r", label='ture liner')
plt.plot(x, np.dot(x, LinearLeastSquareModel_k[:, 0]), label='LinearLeastSquareModel liner')
plt.scatter(best_in_data[:, 0], best_in_data[:, 1], c="r", marker='x', label='in_data')
plt.plot(x, np.dot(x, best_k[:, 0]), c='g', label='Ransac liner')
plt.legend()
plt.show()