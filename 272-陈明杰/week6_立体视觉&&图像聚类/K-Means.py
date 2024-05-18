import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# if __name__ == '__main__':
#     # 读取灰度图
#     img = cv2.imread("lenna.png", 0)
#     src = img
#
#     # 把二维矩阵转化为一维矩阵
#     # print(img)
#     high, width = img.shape
#     img = np.reshape(img, (high * width, 1))
#     # 这里必须要转化为浮点数
#     img = np.float32(img)
#     # print(img)
#
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
#     flags = cv2.KMEANS_RANDOM_CENTERS
#     # 假设 data 是你的数据点数组
#     # K 是聚类的数量
#     # criteria 是停止迭代的条件，通常是一个元组 (type, max_iter, epsilon)
#     # 其中 type 可以是 cv2.TERM_CRITERIA_EPS 或 cv2.TERM_CRITERIA_MAX_ITER 或它们的组合
#     # max_iter 是最大迭代次数
#     # epsilon 是两次迭代之间的相对差异阈值
#     # attempts 是用来重复 kmeans 算法的次数，返回最佳的结果
#     # flags 是用来控制算法的初始化中心的标志，通常是 cv2.KMEANS_RANDOM_CENTERS 或 cv2.KMEANS_PP_CENTERS
#     ret, labels, centers = cv2.kmeans(img, 4, None, criteria, 10, flags)
#     print(ret)
#     img_new = np.reshape(labels, (high, width))
#     print(img_new)
#
#     # 用来正常显示中文标签
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     # 显示图像
#     titles = [u'原始图像', u'聚类图像']
#
#     arr = [src, img_new]
#     for i in range(2):
#         plt.subplot(1, 2, i + 1)
#         plt.imshow(arr[i], 'gray')
#         plt.title(titles[i])
#         plt.xticks()
#         plt.yticks()
#     # plt.subplot(121)
#     # plt.imshow(src,'gray')
#     # plt.title(titles[0])
#     # plt.xticks([])
#     # plt.yticks([])
#     #
#     # plt.subplot(122)
#     # plt.imshow(img_new,'gray')
#     # plt.title(titles[1])
#     # plt.xticks([])
#     # plt.yticks([])
#     plt.show()
#
# print(cv2.__version__)


# if __name__ == '__main__':
#     # 读取图片
#     img = cv2.imread('lenna.png')
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow('gray', gray)
#     # cv2.waitKey(0)
#     # 转化格式，变为列向量
#     high, width = img.shape[:2]
#     src = gray
#     gray = gray.reshape((high * width, 1))
#     # kmeans必须要浮点数格式
#     data = np.float32(gray)
#     # 停止条件
#     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     # 随机质心
#     flags = cv2.KMEANS_RANDOM_CENTERS
#     res, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
#     dst = labels.reshape((high, width))
#     # plt.figure()
#     # plt.subplot(121)
#     # plt.imshow(img)
#     # plt.subplot(122)
#     # plt.imshow(dst, 'gray')
#
#     plt.show()
#     # 显示图像
#     # 用来正常显示中文标签
#     # plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     titles = [u'原始图像', u'聚类图象']
#     images = [src, dst]
#     plt.figure()
#     for i in range(2):
#         plt.subplot(1, 2, 1 + i)
#         plt.imshow(images[i], 'gray')
#         plt.title(titles[i])
#         plt.xticks([])
#         plt.yticks([])
#     plt.show()

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 转换格式
    high, width = gray.shape[:2]
    src = gray.reshape((high * width, 1))
    src = np.float32(src)

    # 停止条件
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # 随机质心
    flags = cv2.KMEANS_RANDOM_CENTERS

    # kmeans聚类
    res, labels, centers = cv2.kmeans(src, 4, None, criteria, 10, flags)
    # 把labels转换为high*width的格式
    res = centers[labels.flatten()]
    print(res)
    dst = res.reshape((high, width))

    # 画图
    plt.figure()
    plt.subplot(121)
    plt.imshow(gray, 'gray')
    plt.title('src')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(dst, 'gray')
    plt.title('dst')
    plt.xticks([])
    plt.yticks([])
    plt.show()
