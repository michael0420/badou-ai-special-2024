import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 定义标准化函数
def normalize(x):
    return (x - np.mean(x)) / np.std(x)


# 归一化的两种方式
def Normalization1(x):
    '''归一化（0~1）'''
    '''x_=(x−x_min)/(x_max−x_min)'''
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]


def Normalization2(x):
    '''归一化（-1~1）'''
    '''x_=(x−x_mean)/(x_max−x_min)'''
    return [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]


#标准化
def z_score(x):
    '''x∗=(x−μ)/σ'''
    x_mean = np.mean(x)
    s2 = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)
    return [(i - x_mean) / s2 for i in x]


# 生成测试数据
x = np.float32(np.random.randint(1, 500, 30))

# 归一化处理
x_normalized1 = normalize(x)
x_normalized2 = normalize(x)
z_result = z_score(x)
# 打印结果,两列结果展示
final_result = pd.DataFrame({'原始数据': x, '归一化1': x_normalized1, '归一化2': x_normalized2, 'z_score': z_result})
print(final_result)
