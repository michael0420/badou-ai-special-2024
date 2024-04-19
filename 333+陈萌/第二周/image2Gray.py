# 对图片进行灰度化和二值化
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray


def image2Gray(img):
    # 获取图像的宽、高、通道
    w,h,c = img.shape
    # 创建一个和该图像相同大小的灰度图像
    img_gray = np.zeros((w,h),dtype=img.dtype)
    for i in range(w):
        for j in range(h):
            # 获取该像素点的RGB值
            b,g,r = img[i,j]
            # 将RGB值转换为灰度值
            img_gray[i,j] = int(0.3*r + 0.59*g + 0.11*b)

    return img_gray

# 灰度化方法一:读取图像的每个像素点的GBR值 然后处理得到该像素点的灰度值
img_1 = cv2.imread('lenna.png') # 整数
img_gray = image2Gray(img_1)
cv2.imshow('img',img_1)
print("img_1:"+str(img_1))
cv2.imshow('img_gray',img_gray)
print("img_gray:"+str(img_gray))

# 等待按键事件 0表示无限期等待 按任意键结束
cv2.waitKey(0)
# 销毁所有窗口
cv2.destroyAllWindows()

# 灰度化方法二 ：直接调用cv2的函数cv2.cvtColor()
img_2 = plt.imread('lenna.png') # 浮点
img_gray2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY) # 取决于img_2的数据类型
print("img_gray2:"+str(img_gray2))
plt.subplot(221)
plt.imshow(img_2,cmap='gray')
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')
plt.subplot(223)
plt.imshow(img_gray2,cmap='gray')

# 灰度化方法三：调用skimage.color的rgb2gray()函数
img_gray3 = rgb2gray(img_2) # 浮点数
print("img_gray3:"+str(img_gray3))
plt.subplot(224)
plt.imshow(img_gray3,cmap='gray')
plt.show()


# 二值化
def img2Binary(img):
    img_binary = np.zeros(img.shape,dtype=img.dtype)
    if img.dtype == 'uint8':
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j] > 127:
                    img_binary[i,j] = 255
                else:
                    img_binary[i,j] = 0
    else:
        img_binary = np.where(img > 0.5,1,0)
    print("img_binary:"+str(img_binary))
    return img_binary

plt.subplot(221)
plt.imshow(img_2)

plt.subplot(222)
plt.imshow(img2Binary(img_gray),cmap='gray')
plt.subplot(223)
plt.imshow(img2Binary(img_gray2),cmap='gray')
plt.subplot(224)
plt.imshow(img2Binary(img_gray3),cmap='gray')
plt.show()