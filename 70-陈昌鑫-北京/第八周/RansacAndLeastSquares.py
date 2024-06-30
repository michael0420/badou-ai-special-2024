import numpy as np
import scipy as sp
import scipy.linalg as sl
import pylab

def ransac(data, model, num, times, th, samples, debug = False, return_all  = False):
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    while iterations < times:
        maybe_idxs, test_idxs = random_partition(num, data.shape[0])
        maybe_inliers = data[maybe_idxs,:]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybe_inliers)
        test_err = model.get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < th]
        also_inliers = data[also_idxs,:]
        if (len(also_inliers) > samples):
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        iterations += 1

    if bestfit is None:
        raise ValueError("Did't meet fit acceptance criteria")
    if return_all:
        return bestfit,{'inliers': best_inlier_idxs}
    else:
        return bestfit

def random_partition(n, n_data):
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

class LesatSquares:
    def __init__(self, input_colums, output_colums, debug = False):
        self.input_colums = input_colums
        self.output_colums = output_colums
        self.debug = debug

    def fit(self, data):
        X = np.vstack( [data[:,i] for i in self.input_colums]).T
        Y = np.vstack( [data[:,i] for i in self.output_colums]).T
        k_vector, resids, rank, s = sl.lstsq(X, Y)
        return k_vector

    def get_error(self, data, model):
        X = np.vstack([data[:, i] for i in self.input_colums]).T
        Y = np.vstack([data[:, i] for i in self.output_colums]).T
        Y_fit = np.dot(X, model)
        err_per_point = np.sum((Y-Y_fit)**2, axis=1)
        return err_per_point

def main():
    iSamples = 500
    iInputs = 1
    iOutputs = 1
    xExact = 20 * np.random.random((iSamples, iInputs)) #随机500个数作用为x
    rate = 60 * np.random.normal(size=(iInputs, iOutputs)) #随机得到一个斜率
    yExact = np.dot(xExact, rate) #得到y

    xNoisy = xExact + np.random.normal(size=xExact.shape) #给x加一个随机值
    yNoisy = yExact + np.random.normal(size=yExact.shape) #给Y随机加个值

    iOutliers = 100
    all_idxs = np.arange(xNoisy.shape[0]) #生成0-400的索引数组
    np.random.shuffle(all_idxs) #随机打乱数组
    outliers_idxs = all_idxs[:iOutliers] #获取前100个
    xNoisy[outliers_idxs] = 20*np.random.random((iOutliers,iInputs)) #噪音点x
    yNoisy[outliers_idxs] = 50 * np.random.normal(size=(iOutliers, iOutputs)) #噪音点Y

    all_data = np.hstack((xNoisy, yNoisy)) #组合成噪音坐标
    input_columns = range(iInputs)
    output_columns = [iInputs + i for i in range(iOutputs)]

    debug = False
    model = LesatSquares(input_columns, output_columns, debug=debug)

    linera_fit, resids, rank, s = sp.linalg.lstsq(all_data[:,input_columns], all_data[:,output_columns])

    ransac_fit, ransac_data = ransac(all_data, model, 50, 1e3, 7e3, 300, debug = False, return_all = True)

    sort_idxs = np.argsort(xExact[:,0]) #将x中的元素从小到大排列，提取其在排列前对应的index(索引)输出
    X_col0_sorted = xExact[sort_idxs]

    pylab.plot(xNoisy[:,0], yNoisy[:,0], 'k.', label = 'data')
    pylab.plot(xNoisy[ransac_data['inliers'], 0], yNoisy[ransac_data['inliers'], 0], 'bx', label = "RANSAC data")

    pylab.plot(X_col0_sorted[:,0], np.dot(X_col0_sorted, ransac_fit)[:,0], label = 'RANSAC fit')
    pylab.plot(X_col0_sorted[:,0], np.dot(X_col0_sorted, rate)[:,0], label = 'Perfect fit')
    pylab.plot(X_col0_sorted[:, 0], np.dot(X_col0_sorted, linera_fit)[:, 0], label='Linera fit')
    pylab.legend()
    pylab.show()

if __name__ == "__main__":
    main()