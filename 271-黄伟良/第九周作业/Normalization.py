import numpy as np

def normalization_guiyi(org, h, w):
    """
    org为二维矩阵
    h:为样本数量-行
    w:特征数-列
    """
    temp_org = np.array(org).T
    nor_org = np.zeros([3, 3], float)
    for i in range(w):
        minValue = float(np.min(temp_org[i]))
        maxValue = float(np.max(temp_org[i]))
        for j in range(h):
            nor_org[i][j] = (temp_org[i][j] - minValue) / (maxValue - minValue)
            print()
    nor_org = nor_org.T

def normalization_z_score(org, h, w, sitar):
    """
    org为二维矩阵
    h:为样本数量-行
    w:特征数-列
    """
    temp_org = np.array(org).T
    nor_org = np.zeros([3, 3], float)
    for i in range(w):
        averValue = float(np.average(temp_org[i]))
        for j in range(h):
            nor_org[i][j] = (temp_org[i][j] - averValue) / sitar
    nor_org = nor_org.T
    print(nor_org)

normalization_guiyi([[1,2,3],[4,5,6],[7,8,9]],3,3)
normalization_z_score([[1,2,3],[4,5,6],[7,8,9]],3,3,1)

