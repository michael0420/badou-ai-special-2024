import numpy as np
import cv2
from collections import Counter
import copy
from matplotlib import pyplot as plt

def histogram_equalization(src_img):
    # 原图像像素点总个数
    pixel = src_img[0].shape[0] * src_img[0].shape[1]
    des_img_list = []
    tmp = np.copy(src_img[0])
    for dimension in range(len(src_img)):
        # 将原图像的二维数组转换为一维数组
        one_dimensional_array = src_img[dimension].flatten().tolist()
        # print(one_dimensional_array)
        # 去重
        one_dimensional_set = set(one_dimensional_array)
        # 求原数组中每个元素的个数并转换为字典
        counts_dict = Counter(one_dimensional_array)
        sum = 0

        # 原像素值与目标像素值的对应关系字典
        fun_dic = {}
        # 循环原图像中所有的像素值
        for src_x in list(one_dimensional_set):
            # 每个像素值的占比 并求和
            sum += counts_dict.get(src_x) / pixel
            # 原像素值转换为目标像素值
            des_x = max(round(sum * 256 - 1), 0)
            fun_dic[src_x] = des_x
        for src_h in range(src_img[dimension].shape[0]):
            for src_w in range(src_img[dimension].shape[1]):
                tmp[src_h, src_w] = fun_dic.get(src_img[dimension][src_h, src_w])
        des_img_list.append(copy.deepcopy(tmp))
    # cv2.imshow("tmp1", des_img_list[0])
    # cv2.imshow("tmp2", des_img_list[1])
    # cv2.imshow("tmp3", des_img_list[2])
    return cv2.merge((des_img_list[0], des_img_list[1], des_img_list[2]))

src_img = cv2.imread("lenna.png", )

gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()
cv2.imshow("Gray Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)

# 彩色图像直方图均衡化
(b, g, r) = cv2.split(src_img)
des_img = np.zeros((src_img.shape[0], src_img.shape[1], src_img.shape[2]), np.uint8)

des_img = histogram_equalization((b, g, r))
cv2.imshow("Color Histogram Equalization", np.hstack([src_img, des_img]))
cv2.waitKey(0)