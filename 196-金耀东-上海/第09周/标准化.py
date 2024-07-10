import numpy as np
import matplotlib.pyplot as plt

def normalize1(data):
    ''' 归一化到[0，1] '''
    return ( data - data.min() ) / ( data.min() - data.max() )

def normalize0(data):
    ''' 0均值归一化 '''
    return ( data - data.mean() ) / ( data.min() - data.max() )

def z_score(data):
    ''' x* = (x−μ)/σ '''
    return ( data - data.mean() ) / data.std()

if __name__ == "__main__":
    # 测试数据
    src_data = np.array( [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
         11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30] , dtype=int)

    # 归一化[0,1]
    normalized1_data = normalize1(src_data)

    # 0均值归一化
    normalized0_data = normalize0(src_data)

    # z-score标准化
    z_data = z_score(src_data)

    # 将数据个数作为Y轴
    y = [np.sum(src_data==i) for i in src_data]

    # 展示结果
    plt.plot(src_data, y)
    plt.plot(normalized1_data, y)
    plt.plot(normalized0_data, y)
    plt.plot(z_data, y)
    plt.show()

    pass