import cv2
import numpy as np
import time
import os.path as path
def aHash(img, width=8, high=8):

    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    avg = s / 64
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def dHash(img, width=9, high=8):

    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(high):
        for j in range(high):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def cmp_hash(hash1, hash2):

    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1

    return 1 - n / len(hash2)


def pHash(img_file, width=64, high=64):

    img = cv2.imread(img_file, 0)
    img = cv2.resize(img, (width, high), interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img
    vis1 = cv2.dct(cv2.dct(vis0))
    vis1.resize(32, 32)
    img_list = vis1.flatten()
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = ['0' if i > avg else '1' for i in img_list]
    return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 32 * 32, 4)])

def hamming_dist(s1, s2):
    return 1 - sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)]) * 1. / (32 * 32 / 4)

def concat_info(type_str, score, time):
    temp = '%s相似度：%.2f %% -----time=%.4f ms' % (type_str, score * 100, time)
    print(temp)
    return temp

def test_diff_hash(img1_path, img2_path, loops=1000):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    start_time = time.time()

    for _ in range(loops):
        hash1 = dHash(img1)
        hash2 = dHash(img2)
        cmp_hash(hash1, hash2)

    print(">>> 执行%s次耗费的时间为%.4f s." % (loops, time.time() - start_time))


def test_aHash(img1, img2):
    time1 = time.time()
    hash1 = aHash(img1)
    hash2 = aHash(img2)
    n = cmp_hash(hash1, hash2)
    return concat_info("均值哈希算法", n, time.time() - time1) + "\n"

def test_dHash(img1, img2):
    time1 = time.time()
    hash1 = dHash(img1)
    hash2 = dHash(img2)
    n = cmp_hash(hash1, hash2)
    return concat_info("差值哈希算法", n, time.time() - time1) + "\n"

def test_pHash(img1_path, img2_path):
    time1 = time.time()
    hash1 = pHash(img1_path)
    hash2 = pHash(img2_path)
    n = hamming_dist(hash1, hash2)
    return concat_info("感知哈希算法", n, time.time() - time1) + "\n"

def deal(img1_path, img2_path):
    info = ''

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 计算图像哈希相似度
    info = info + test_aHash(img1, img2)
    info = info + test_dHash(img1, img2)
    info = info + test_pHash(img1_path, img2_path)
    return info

def contact_path(file_name):
    output_path = "./source"
    return path.join(output_path, file_name)

def main():
    data_img_name = 'lenna.png'
    data_img_name_base = data_img_name.split(".")[0]

    base = contact_path(data_img_name)
    light = contact_path("%s_light.jpg" % data_img_name_base)
    resize = contact_path("%s_resize.jpg" % data_img_name_base)
    contrast = contact_path("%s_contrast.jpg" % data_img_name_base)
    sharp = contact_path("%s_sharp.jpg" % data_img_name_base)
    blur = contact_path("%s_blur.jpg" % data_img_name_base)
    color = contact_path("%s_color.jpg" % data_img_name_base)
    rotate = contact_path("%s_rotate.jpg" % data_img_name_base)

    # 测试算法的效率
    test_diff_hash(base, base)
    test_diff_hash(base, light)
    test_diff_hash(base, resize)
    test_diff_hash(base, contrast)
    test_diff_hash(base, sharp)
    test_diff_hash(base, blur)
    test_diff_hash(base, color)
    test_diff_hash(base, rotate)

    # 测试算法的精度(以base和light为例)
    deal(base, light)


if __name__ == '__main__':
    main()

import numpy as np
import scipy as sp
import scipy.linalg as sl


def ransac(data, model, n, k, t, d, debug=False, return_all=False):

    iterations = 0
    bestfit = None
    besterr = np.inf  # 设置默认值
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        print('test_idxs = ', test_idxs)
        maybe_inliers = data[maybe_idxs, :]  # 获取size(maybe_idxs)行数据(Xi,Yi)
        test_points = data[test_idxs]  # 若干行(Xi,Yi)数据点
        maybemodel = model.fit(maybe_inliers)  # 拟合模型
        test_err = model.get_error(test_points, maybemodel)  # 计算误差:平方和最小
        print('test_err = ', test_err < t)
        also_idxs = test_idxs[test_err < t]
        print('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs, :]
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', numpy.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        # if len(also_inliers > d):
        print('d = ', d)
        if (len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
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
        B_fit = sp.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point

def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    B_exact = sp.dot(A_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        # 添加"局外点"
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idxs)  # 将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi
    # setup model
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型

    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
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


