'''
直方图均衡化
'''

import numpy as np
import cv2

# 均衡化
def function(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width, channels = img.shape
    pixAry = []
    nixDict = {}
    for i in range(height):
        for j in range(width):
            # 数据分类
            value = img_gray[i, j]
            if value in pixAry:
                # 元素已存在，则 +1
                pixV = int(nixDict[str(value)])
                nixDict[str(value)] = str(pixV + 1)
            else:
                # 元素不存在, 则存储
                nixDict[str(value)] = str(value)
                pixAry.append(value)

    # 按照大小进行过排序
    sorted_arr = sorted(pixAry)
    # 图片像素总量
    totality = height * width
    # 用于保存 pi
    piniDict = {}
    # 算出每个 pix 的pi
    for i in range(len(sorted_arr)):
        pix = sorted_arr[i]
        # 取出元素总个数
        niV = int(nixDict[str(pix)])
        # 计算 pi=ni/image
        piniV = niV / totality
        piniDict[str(pix)] = piniV

    sumPi = 0
    emptyImage = np.zeros((height, width, channels), np.uint8)
    for i in range(height):
        for j in range(width):
            locV = img_gray[i, j]
            pi = piniDict[str(locV)]
            sumPi+=pi
            locV_new =int(sumPi*256-1+0.5) #加0.5取整，和四舍五入一样
            emptyImage[i, j] = locV
    return emptyImage

img = cv2.imread('lenna.png')
zoom = function(img)
cv2.imshow('histogram equalization', zoom)
cv2.waitKey()





