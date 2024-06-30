import numpy as np
import scipy as sp
import scipy.linalg as sl

if __name__ == '__main__':
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:列向量 （500,1）
    perfect_fit = 60 * np.random.normal(
        size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率 用于生成符合正态分布（也称为高斯分布）的随机数 loc 表示均值 scale表示标准差
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

    A = np.vstack(all_data[:, 0])
    B = np.vstack(all_data[:, 1])

    x, residues, rank, s = sl.lstsq(A, B)

    all_idxs = np.arange(all_data.shape[0])  # 获取n_data下标索引
    np.random.shuffle(all_idxs)  # 打乱下标索引
    idxs1 = all_idxs[:50]
    idxs2 = all_idxs[50:]

    maybe_inliers = all_data[idxs1, :]
    test_points = all_data[idxs2]
