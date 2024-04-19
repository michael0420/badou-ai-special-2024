"""
直方图均衡化
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


# 计算图片的灰度值直方图
def cal_hist(img):
    h, w, c = img.shape
    hist = np.zeros((256, c))
    for i in range(h):
        for j in range(w):
            for k in range(c):
                hist[img[i, j, k], k] += 1
    return hist


def equalizeHist(img):
    h, w, c = img.shape
    out = np.zeros((h, w, c), dtype=np.uint8)
    hist = cal_hist(img)
    # 对每个值进行累计
    hist = np.cumsum(hist, axis=0)
    for i in range(h):
        for j in range(w):
            for k in range(c):
                out[i, j, k] = hist[img[i, j, k], k] * 256 / (h * w) - 1
                if out[i, j, k] < 0:
                    out[i, j, k] = 0

    return out


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    # hist = cal_hist(img)
    # api调用计算直方图
    # (256,256,256)
    # hist = cv2.calcHist([img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    # b通道 只能单独求
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    print(hist)
    # print(equalizeHist(img))

    # 灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # 将二维数据扩展成三维数组
    # img = np.expand_dims(img_gray, axis=2)

    # print(img.shape)
    # dst_img = equalizeHist(img)

    # 灰度图
    # dst_img = cv2.equalizeHist(img_gray)
    # plt.figure() # 创建一个窗口
    # # img_gray.ravel() 展成一维数组 plt.hist 直方图
    # plt.subplot(221), plt.hist(img_gray.ravel(), bins=256), plt.title('Original Image')
    # plt.subplot(222), plt.hist(dst_img.ravel(), bins=256), plt.title('Destination Image')
    # plt.show()

    # 彩色图
    b,g,r = cv2.split(img) # 拆分通道 分解成三个数组
    print(b.shape)
    # 分别对三个通道进行均衡化
    be = cv2.equalizeHist(b)
    ge = cv2.equalizeHist(g)
    re = cv2.equalizeHist(r)
    # 在一个直方图上绘制三个通道的直方图
    plt.subplot(221), plt.hist(b.ravel(), bins=256,color='blue',label='b',alpha=0.5),
    plt.hist(g.ravel(), bins=256, color='green',label='g',alpha=0.5),plt.hist(r.ravel(), bins=256,color='red',label='r',alpha=0.5),
    plt.title('Original Image')
    plt.legend()

    plt.subplot(222), plt.hist(be.ravel(), bins=256, color='blue', label='b', alpha=0.5),
    plt.hist(ge.ravel(), bins=256, color='green', label='g', alpha=0.5), plt.hist(re.ravel(), bins=256, color='red',
                                                                                 label='r', alpha=0.5),
    plt.title('Destination Image')
    plt.legend()
    # plt.show()

    # 对三个通道进行合并
    dst_img=cv2.merge((be,ge,re))

    print(img.ravel().shape)
    plt.subplot(223), plt.hist(img.ravel(), bins=256), plt.title('img')
    # plt.show()
    plt.subplot(224), plt.hist(dst_img.ravel(), bins=256), plt.title('dst_img')
    plt.show()
    cv2.imshow('img', np.hstack((img, dst_img)))
    cv2.imshow('gray_img', np.hstack((img_gray, cv2.equalizeHist(img_gray))))
    cv2.waitKey(0)
