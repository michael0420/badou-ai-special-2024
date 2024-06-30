import numpy as np
import matplotlib.pyplot as plt

# 生成数据
'''
num_points 正常数据数
outliers_ratio 异常数据比例
noise_sigma 正常数据添加高斯噪声sigma
'''
def generate_data(num_points, outliers_ratio=0.3, noise_sigma=1.0):
    np.random.seed(13)
    # 正常数据
    x = np.random.randint(0, 500, num_points)
    y = 3.5 * x + np.random.normal(0, noise_sigma, num_points)
    # 异常数据
    outliers_num = int(num_points * outliers_ratio)
    x_outliers = np.random.uniform(0, 500, outliers_num)
    y_outliers = np.random.uniform(0, 3.5*500, outliers_num)

    x = np.concatenate([x, x_outliers])
    y = np.concatenate([y, y_outliers])
    return x, y


'''
x,y 数据
n 随机选择几个点设定为内群
num_iterations 迭代次数
threshold 阈值
'''
def ransac(x, y, n = 2, num_iterations=100, threshold=1.0):
    '''
    best_model：存储当前最佳模型的参数（斜率和截距）。
    best_inliers：存储当前最佳模型的内点数量。
    best_inlier_indices：存储当前最佳模型的内点索引。
    '''
    best_model = None
    best_inliers = 0
    best_inlier_indices = None

    for _ in range(num_iterations):
        # 随机选择n个点来拟合模型
        sample_indices = np.random.choice(len(x), n, replace=False)
        x_sample = x[sample_indices]
        y_sample = y[sample_indices]

        # 拟合线性模型
        A = np.vstack([x_sample, np.ones(len(x_sample))]).T
        m, c = np.linalg.lstsq(A, y_sample, rcond=None)[0]

        # 计算所有点到模型的距离
        y_pred = m * x + c
        r = np.abs(y - y_pred)

        # 统计内点数量
        inliers = r < threshold
        num_inliers = np.sum(inliers)

        # 更新最佳模型
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_model = (m, c)
            best_inlier_indices = np.where(inliers)[0]

    print("best_inliers:", best_inliers)
    return best_model, best_inlier_indices


# 4.生成图表

if __name__ == '__main__':
    x, y = generate_data(100, outliers_ratio=0.8, noise_sigma=20.0)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c='blue', marker='o', label='Data points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('ransac')
    plt.legend()
    plt.grid(True)

    best_model, best_inlier_indices = ransac(x, y, n=5, threshold=20.0)
    k,b = best_model
    x_vals = np.linspace(min(x), max(x), 100)
    y_vals = k * x_vals + b
    plt.plot(x_vals, y_vals, color='green', label=f'y = {k}x + {b}')
    plt.legend()

    # 使用最小二乘法再画一条线
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    x_vals2 = np.linspace(min(x), max(x), 100)
    y_vals2 = m * x_vals2 + c
    plt.plot(x_vals2, y_vals2, color='yellow', label=f'y = {m}x + {c}')
    plt.legend()


    plt.show()