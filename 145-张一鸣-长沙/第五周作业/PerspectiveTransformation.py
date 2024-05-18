# coding = utf-8

'''
        实现图像透视变换
'''

import cv2
import numpy as np

# 调用cv2接口实现透视变换
# src = cv2.imread('photo1.jpg')
ori = cv2.imread('yy1.jpg')
# cv2.imshow('src', src)
cv2.imshow('ori', ori)
#
# result_1 = src.copy()
result_2 = ori.copy()
#
# ori_1 = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
# new_1 = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
#
# ori_2 = np.float32([[27, 48], [521, 46], [26, 299], [523, 299]])
# new_2 = np.float32([[0, 0], [400, 0], [0, 300], [400, 300]])
#
# warp_1 = cv2.getPerspectiveTransform(ori_1, new_1)          # 生成变换矩阵
# result = cv2.warpPerspective(result_1, warp_1, (337, 488))  # 产生新图像
#
# warp_2 = cv2.getPerspectiveTransform(ori_2, new_2)
# dst = cv2.warpPerspective(result_2, warp_2, (400, 300))
#
# print(warp_1)
# print(warp_2)
# cv2.imshow('warp1', result)
# cv2.imshow('warp2', dst)
# cv2.waitKey(0)

# 手动求warpMatrix实现透视变换
def warpMatrix(ori_3, new_3):
    # 断言原始点与目标点数量是否一致，且至少需要4组
    assert ori_3.shape[0] == new_3.shape[0] and ori_3.shape[0] >= 4
    w = ori_3.shape[0]
    A = np.zeros((w*2, 8))      # A矩阵8列
    B = np.zeros((w*2, 1))      # B矩阵1列
    for i in range(0, w):
        A_i = ori_3[i, :]
        B_i = new_3[i, :]
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[2*1] = B_i[0]
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]
    # 将A从ndarray类型转换为matrix类型，允许使用矩阵乘法运算符*和逆矩阵属性A.I
    A = np.mat(A)
    det_A = np.linalg.det(A)    # 计算行列式可判断是否为奇异矩阵
    print('A的行列式为:\n', det_A)
    if det_A != 0:
        # 求A的逆矩阵，A.I * B = warpMatrix，求出8个未知数
        warp_3 = A.I * B
        print('计算出的warpMatrix:\n', warp_3)
        # warpMatrix的处理
        warp_3 = np.array(warp_3).T[0]
        warp_3 = np.insert(warp_3, warp_3.shape[0], values=1.0, axis=0)     # 插入a33=1
        warp_3 = warp_3.reshape((3, 3))
        return warp_3
    else:
        return 'A为奇异矩阵，没有逆矩阵'



if __name__ == '__main__':
    ori_3 = [[27.0, 48.0], [521.0, 46.0], [26.0, 299.0], [523.0, 299.0]]
    new_3 = [[0.0, 0.0], [400.0, 0.0], [0.0, 300.0], [400.0, 300.0]]
    ori_3 = np.array(ori_3)
    new_3 = np.array(new_3)

    warp = warpMatrix(ori_3, new_3)
    print('最终的warpMatrix:\n', warp)

    dst = cv2.warpPerspective(result_2, warp, (400, 300))
    cv2.imshow('warp2', dst)
    cv2.waitKey(0)
