import cv2
import numpy as np
import matplotlib.pyplot as plt

# 图像归一化
def toNormalize(img, max_value=255.0):
    return img / max_value

# 图像反归一化
def inverseNormalize(normalied_img, max_value=255.0):
    return (normalied_img * max_value).astype(np.uint8)

# 像素灰度化函数
def convertToGray(pixel):
    return pixel[0] * 0.3 + pixel[1] * 0.59 + pixel[2] * 0.11

# 像素二值化函数
def convertToBinary(pixel):
    return round(pixel)

# 将源图像转化为目标图像
def convertImg(src_img, convert_func):
    height, width = src_img.shape[:2]
    dst_img = np.zeros([height, width], dtype=float)
    tmp_img = toNormalize( src_img )
    for i in range(height):
        for j in range(width):
            dst_img[i, j] = convert_func( tmp_img[i, j] )
    return inverseNormalize(dst_img)

# 读取RGB图
def readRGBImg(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

# RGB图灰度化
def rgbToGray(img):
    return convertImg(img, convert_func=convertToGray)

# 灰度图二值化
def grayToBinary(img):
    return convertImg(img, convert_func=convertToBinary)

# 将所有图像展示在一个平面图中
def displayImgs(imgs, titles, row, col):
    for i in range(len(imgs)):
        plt.subplot(row, col, i + 1)
        plt.imshow(imgs[i], cmap="gray")
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # 不显示横纵坐标轴
    plt.show()

if __name__ == "__main__":
    img_path = "img/lenna.jpg"

    rgb_img = readRGBImg(img_path)
    gray_img = rgbToGray(rgb_img)
    bin_img = grayToBinary(gray_img)

    displayImgs(
        imgs = [rgb_img, gray_img, bin_img],
        titles = ["RGB", "Gray", "Binary"],
        row = 1, col = 3
    )

