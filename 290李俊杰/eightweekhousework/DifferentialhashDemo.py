'''

实现差值哈希

'''

import cv2
import numpy as np
def blur(image):
    # 模糊操作
    return cv2.blur(image, (15, 1))


def sharp(image):
    # 锐化操作
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    return cv2.filter2D(image, -1, kernel=kernel)

def diffhash(img):
    # s缩放图片8*9
    img=cv2.resize(img,(8,9),interpolation=cv2.INTER_CUBIC)
    # 灰度化
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 比较，像素值大于后面一个像素值记作1，小于记作0，本行不与下一行比较
    # 每行生成八个差值1,0，一共64位指纹
    hashstr=''
    h,w=img_gray.shape
    for i in range(h):
        for j in range(w):
            if j+1<=w-1:
                if img_gray[i][j]>=img_gray[i][j+1]:
                    hashstr+='1'
                else:
                    hashstr += '0'
    return hashstr

img=cv2.imread("lenna.png")
# czhash=diffhash(img)
# print(czhash)

# 生成两张新图使用均值哈希进行对比
img_blur=blur(img)
averhash_blur=diffhash(img_blur)
print(averhash_blur)

img_sharp=sharp(img)
averhash_sharp=diffhash(img_sharp)
print(averhash_sharp)
# 对比两张新图计算汉明距离即有多少位上的1,0不一样，不同位数越少，相似度越高
different_num=0
for i in range(len(averhash_blur)):
    if averhash_blur[i]!= averhash_sharp[i]:
        different_num+=1
print(different_num)