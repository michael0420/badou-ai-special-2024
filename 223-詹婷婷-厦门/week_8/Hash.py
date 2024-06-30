import cv2
import numpy as np

def aHash(img):
    #缩放为8*8
    """
    def resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None):
    src:输入图像
    dsize:输出图像的大小（宽，高）
    fx:width方向的缩放比例
    fy:height方向的缩放比例
    interpolation:插值方式，默认为双线性插值
        INTER_NEAREST:最邻近插值
        INTER_LINEAR:双线性插值 （默认）
        INTER_CUBIC:4x4像素邻域内的双立方插值
        INTER_AREA:使用像素区域关系进行重采样
        INTER_LANCZOS4:8x8像素邻域内的Lanczos插值
    """
    img = cv2.resize(img, (8,8), interpolation=cv2.INTER_CUBIC)

    #转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sum = 0
    hasg_str = ''
    for i in range(8):
        for j in range(8):
            sum += gray[i,j]

    avg = sum / 64
    print(avg)
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                hasg_str = hasg_str+'1'
            else:
                hasg_str = hasg_str+'0'
    return hasg_str

def dHash(img):
    img = cv2.resize(img, (9,8), interpolation=cv2.INTER_CUBIC)
    #转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hasg_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i][j] > gray[i][j+1]:
                hasg_str = hasg_str + '1'
            else:
                hasg_str = hasg_str + '0'
    return hasg_str

#Hash值对比
def cmpHash(hash1, hash2):
    n = 0
    #hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        if hash1[i] != hash2[2]:
            n += 1
    return n



img1 = cv2.imread("lenna.png")
img2 = cv2.imread("lenna_blur.jpg")

img1_aHash = aHash(img1)
img2_aHash = aHash(img2)

print(img1_aHash)
print(img2_aHash)

n = cmpHash(img1_aHash, img2_aHash)
print(n)


img1_dHash = dHash(img1)
img2_dHash = dHash(img2)
print(img1_dHash)
print(img2_dHash)

m = cmpHash(img1_dHash, img2_dHash)
print(m)