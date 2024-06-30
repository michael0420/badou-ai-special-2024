# import numpy as np
# import cv2
#
# 1、
# img = cv2.imread("photo1.jpg")
#
# # 需要抠图的四个原始点的像素坐标
# src_index=np.float32([[204, 155], [519, 285], [16, 605], [344, 731]])
# # 四个原始坐标对应的四个目标坐标，即自己想把图片放到哪里
# dst_index=np.float32([[0, 0], [315, 0], [0, 450], [315, 450]])
#
# # 通过四个原始坐标和四个目的坐标就能求出透视变换矩阵
# Perspective_Transformation_Matrix=cv2.getPerspectiveTransform(src_index,dst_index)
# copy=img.copy()
# # 通过透视变换矩阵就能把整张图片的所有点都变换为目的坐标
# result=cv2.warpPerspective(copy,Perspective_Transformation_Matrix,(315,450))
# cv2.imshow("src",img)
# cv2.imshow("result",result)
# cv2.waitKey(0)
#

# 2、
# # 直接调用库函数
# img = cv2.imread('photo1.jpg')
#
# result3 = img.copy()
#
# '''
# 注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
# '''
# src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
# dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
# print(img.shape)
# # 生成透视变换矩阵；进行透视变换
# m = cv2.getPerspectiveTransform(src, dst)
# print("warpMatrix:")
# print(m)
# result = cv2.warpPerspective(result3, m, (337, 488))
# cv2.imshow("src", img)
# cv2.imshow("result", result)
# cv2.waitKey(0)


import cv2
import numpy as np


# # 3、自己实现求透视变换矩阵的函数
# def getPerspectiveTransform(src, dst):
#     assert (src.shape[0] == dst.shape[0] and src.shape[1] == dst.shape[1] and src.shape[0] >= 4)
#     nums = src.shape[0]
#     # 创建8*8和8*1的数组，根据PPT的表格可知，8*8是A矩阵，8*1是B矩阵
#     A = np.zeros((nums * 2, 8))
#     B = np.zeros((nums * 2, 1))
#
#     for i in range(nums):
#         # 取出每一对原始点对应的目标点，一一对应(x0,y0)->(x1,y1)
#         A_i = src[i, :]
#         B_i = dst[i, :]
#         # 按照表格分别填写偶数行和奇数行的A矩阵和B矩阵对应位置的系数，这些点都是已知的
#         A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
#         B[2 * i, :] = B_i[0]
#         A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
#         B[2 * i + 1, :] = B_i[1]
#
#     # A本身是一个二维数组，要把它转为矩阵的形式才能求逆矩阵
#     A = np.mat(A)
#     # 因为根据PPT的表格可知, A*W=B ，所以A^(-1)*A*W=A^(-1)*B，所以W=A^(-1)*B
#     w = A.I * B
#     # 因为w是矩阵，所以要变成数组的形式，因为w本身是一列的矩阵，所以变成数组之后
#     # 要取转置才能得到一个一维数组
#     w = np.array(w).T[0]
#     # 在数组的最后插入刚刚没有计算的a33,a33=1，为什么a33=1,主要是方便计算，等于别的值也行
#     w = np.insert(w, len(w), 1)
#     # 最后把一个一维的数组变成3*3的数组，那么这个3*3的二维数组就是透视转换矩阵了
#     w = w.reshape((3, 3))
#     # print(w)
#     return w
#
# if __name__ == '__main__':
#     src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
#     dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
#     m = getPerspectiveTransform(src, dst)
#     print(m)

def getPerspectiveTransform(src, dst):
    assert (src.shape == dst.shape and src.shape[0] >= 4)
    num = src.shape[0]
    A = np.zeros([num * 2, 8])
    B = np.zeros([num * 2, 1])
    # print(A)
    # print(B)
    for i in range(num):
        A_i = src[i, :]
        B_i = dst[i, :]
        # print(A_i)
        # print(B_i)
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i, :] = [B_i[0]]
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1, :] = [B_i[1]]

    A = np.mat(A)
    w = A.I * B
    m = np.array(w)
    m=m.T
    m = np.append(m, 1)
    m = m.reshape(3, 3)
    return m


if __name__ == "__main__":
    # src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    # dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    # # 求出透视变换矩阵
    # m = getPerspectiveTransform(src, dst)
    # print(m)
    img = cv2.imread("photo1.jpg")

    # 需要抠图的四个原始点的像素坐标
    src_index=np.float32([[204, 155], [519, 285], [16, 605], [344, 731]])
    # 四个原始坐标对应的四个目标坐标，即自己想把图片放到哪里
    dst_index=np.float32([[0, 0], [315, 0], [0, 450], [315, 450]])

    # 通过四个原始坐标和四个目的坐标就能求出透视变换矩阵
    Perspective_Transformation_Matrix=getPerspectiveTransform(src_index,dst_index)
    copy=img.copy()
    # 通过透视变换矩阵就能把整张图片的所有点都变换为目的坐标
    result=cv2.warpPerspective(copy,Perspective_Transformation_Matrix,(315,450))
    cv2.imshow("src",img)
    cv2.imshow("result",result)
    cv2.waitKey(0)

