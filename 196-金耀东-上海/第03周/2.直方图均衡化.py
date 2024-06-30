import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 最大灰度级
MAX_LEVEL = 255

# 获取单个通道的直方图
def _get_single_hist(img):
    hist = np.zeros(shape=(MAX_LEVEL + 1), dtype=int)
    for pixel in img.ravel():
        hist[ pixel ] += 1
    return hist

# 对单个通道进行直方图均衡化
def _single_to_hist_equalize(img):
    height, width = img.shape[:2]

    # 计算直方图
    hist = _get_single_hist(img)

    # 直方图归一化
    normalized_hist = hist / (height * width)

    # 计算累计直方图
    hist_cumsum = np.cumsum(normalized_hist)

    # 像素映射
    mapping =np.zeros(shape=(MAX_LEVEL + 1), dtype=np.uint8)
    for k in range(MAX_LEVEL + 1):
        mapping[k] = hist_cumsum[k] * MAX_LEVEL

    # 计算映射后的图像
    eq_img = np.zeros(shape=(height, width), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            eq_img[h, w] = mapping[ img[h, w] ]

    return eq_img

# 对灰度图进行直方图均衡化
def gray_to_hist_equalize(img):
    return _single_to_hist_equalize(img)

# 对rgb图像进行直方图均衡化
def rgb_to_hist_equalize(img):
    r = _single_to_hist_equalize( img[:, :, 0] )
    g = _single_to_hist_equalize( img[:, :, 1] )
    b = _single_to_hist_equalize( img[:, :, 2] )
    return cv2.merge( (r, g, b) )

# 展示图像
def display_imgs(imgs, titles, types, rows, cols):
    for i in range( len(imgs) ):
        plt.subplot(rows, cols, i+1)
        plt.imshow(imgs[i], cmap=types[i])
        plt.title(titles[i])
        plt.xticks([]) , plt.yticks([]) # 不显示横纵坐标轴
    plt.show()

if __name__ == "__main__" :
    img_dir = "img"
    img_filename = "lenna.jpg"
    img_path = os.path.join(img_dir, img_filename)

    # 读取bgr图
    bgr_img = cv2.imread(img_path)

    # 获取灰度图与rgb图
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # 计算均衡化后的图像
    gray_eq_img = gray_to_hist_equalize( gray_img )
    rgb_eq_img = rgb_to_hist_equalize( rgb_img )

    # 展示图像
    display_imgs(
        imgs = [rgb_img, rgb_eq_img, gray_img, gray_eq_img],
        titles=["rgb image", "equalized rgb image", "gray image", "equalized gray image"],
        types=[None, None, "gray", "gray"],
        rows=2, cols=2
    )
