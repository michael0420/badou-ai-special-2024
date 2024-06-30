import numpy as np
from numpy.linalg import lstsq
from typing import Tuple, Optional, Dict, Any


def ransac(data: np.ndarray, model: Any, min_samples: int, max_iterations: int, threshold: float,
           min_inliers: int, debug: bool = False, return_all: bool = False) -> Tuple[
    np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """
    实现RANSAC算法。

    参数：
        data (np.ndarray): 样本点。
        model (object): 要拟合数据的假设模型。
        min_samples (int): 拟合模型所需的最少样本数量。
        max_iterations (int): 最大迭代次数。
        threshold (float): 判断点是否符合模型的阈值。
        min_inliers (int): 有效模型所需的最少内点数。
        debug (bool): 如果为True，打印调试信息。
        return_all (bool): 如果为True，返回所有与拟合过程相关的数据。

    返回值：
        Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]: 最佳拟合模型参数和包含内点的字典（如果return_all为True）。

    抛出：
        ValueError: 如果未找到可接受的模型。
    """
    best_model = None
    lowest_error = np.inf
    best_inliers = None

    for iteration in range(max_iterations):
        # 随机分割样本
        sample_indices, test_indices = random_partition(min_samples, data.shape[0])
        sample_data = data[sample_indices]
        test_data = data[test_indices]

        # 拟合样本数据
        trial_model = model.fit(sample_data)
        test_errors = model.get_error(test_data, trial_model)

        # 选择符合阈值的内点
        inliers_mask = test_errors < threshold
        inliers_data = test_data[inliers_mask]

        if debug:
            print(f"迭代 {iteration}: 找到 {len(inliers_data)} 个内点。")

        # 检查内点数量并计算误差
        if len(inliers_data) > min_inliers:
            all_inliers_data = np.vstack((sample_data, inliers_data))
            refined_model = model.fit(all_inliers_data)
            mean_error = np.mean(model.get_error(all_inliers_data, refined_model))

            # 更新最佳模型
            if mean_error < lowest_error:
                best_model = refined_model
                lowest_error = mean_error
                best_inliers = np.concatenate((sample_indices, test_indices[inliers_mask]))

    if best_model is None:
        raise ValueError("未能满足拟合接受标准。")

    if return_all:
        return best_model, {'inliers': best_inliers}
    else:
        return best_model


def random_partition(min_samples: int, total_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    随机将索引分成两组：一组有min_samples，另一组有剩余的样本。

    参数：
        min_samples (int): 第一组所需样本数量。
        total_samples (int): 总样本数量。

    返回值：
        Tuple[np.ndarray, np.ndarray]: 两组索引数组。
    """
    indices = np.random.permutation(total_samples)
    return indices[:min_samples], indices[min_samples:]


class LinearLeastSquaresModel:
    def __init__(self, input_columns: np.ndarray, output_columns: np.ndarray, debug: bool = False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data: np.ndarray) -> np.ndarray:
        A = data[:, self.input_columns]
        B = data[:, self.output_columns]
        model, _, _, _ = lstsq(A, B, rcond=None)
        return model

    def get_error(self, data: np.ndarray, model: np.ndarray) -> np.ndarray:
        A = data[:, self.input_columns]
        B = data[:, self.output_columns]
        B_predicted = A @ model
        return np.sum((B - B_predicted) ** 2, axis=1)


def test():
    np.random.seed(42)
    n_samples = 500  # 正常样本数量
    n_inputs = 1
    n_outputs = 1

    # 生成精确数据和带噪声的数据
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 精确矩阵A
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
    B_exact = A_exact @ perfect_fit  # 点乘

    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    # 添加异常数据
    n_outliers = 100  # 异常样本数量
    outlier_indices = np.random.choice(np.arange(A_noisy.shape[0]), n_outliers, replace=False)
    A_noisy[outlier_indices] = 20 * np.random.random((n_outliers, n_inputs))  # 均匀分布的噪声
    B_noisy[outlier_indices] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 正态分布的噪声

    # 整合所有数据
    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = np.arange(n_inputs)
    output_columns = np.arange(n_inputs, n_inputs + n_outputs)

    # 最小二乘法拟合
    linear_fit, _, _, _ = lstsq(all_data[:, input_columns], all_data[:, output_columns], rcond=None)

    # 使用RANSAC拟合
    model = LinearLeastSquaresModel(input_columns, output_columns, debug=False)
    ransac_fit, ransac_data = ransac(data=all_data, model=model, min_samples=50, max_iterations=1000, threshold=7e3, min_inliers=300, debug=False, return_all=True)

    # 绘制结果
    import matplotlib.pyplot as plt

    sorted_indices = np.argsort(A_exact[:, 0])
    A_sorted = A_exact[sorted_indices]

    plt.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
    plt.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
    plt.plot(A_sorted[:, 0], A_sorted @ ransac_fit, label='RANSAC fit')
    plt.plot(A_sorted[:, 0], A_sorted @ perfect_fit, label='exact system')
    plt.plot(A_sorted[:, 0], A_sorted @ linear_fit, label='linear fit')
    plt.legend()  # 图例
    plt.show()


if __name__ == "__main__":
    test()
