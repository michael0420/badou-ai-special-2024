'''
实现均值哈希
'''

import numpy as np
import cv2



def averagehash(img):
    # 缩放为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 转化成灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 求平均值，计算灰度图所有像素的平均值
    gray_num = 0
    h, w = img_gray.shape
    # print(h,w)
    for i in range(h):
        for j in range(w):
            gray_num += img_gray[i][j]
    average_gray = gray_num // (h * w)
    # print(average_gray)
    # 比较，像素值大于平均值记作1，小于平均值记作0
    # 生成均值hash值，将比较出来的1和0按顺序组合起来就是图片的指纹
    gray_str = ''
    for i in range(h):
        for j in range(w):
            if img_gray[i][j] >= average_gray:
                gray_str += '1'
            else:
                gray_str += '0'
    # print(gray_str)
    return gray_str
img = cv2.imread("lenna.png")
averhash=averagehash(img)
# print(averhash)

def blur(image):
    # 模糊操作
    return cv2.blur(image, (15, 1))


def sharp(image):
    # 锐化操作
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    return cv2.filter2D(image, -1, kernel=kernel)

# 生成两张新图使用均值哈希进行对比
img_blur=blur(img)
averhash_blur=averagehash(img_blur)
print(averhash_blur)
img_sharp=sharp(img)
averhash_sharp=averagehash(img_sharp)
print(averhash_sharp)
# 对比两张新图计算汉明距离即有多少位上的1,0不一样，不同位数越少，相似度越高
different_num=0
for i in range(len(averhash_blur)):
    if averhash_blur[i]!= averhash_sharp[i]:
        different_num+=1

print(different_num)