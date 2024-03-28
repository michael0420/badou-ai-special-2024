
import cv2
import matplotlib.pyplot as plt
import numpy as np

from Util import cv_imread

def near_insert_01(img, new_h, new_w):
    """
    最邻近插值，自己写方法
    new_h: 新图像的高
    new_w: 新图像的宽
    """
    # 获取图片的high和wide
    h, w ,c= img.shape
    # 创建一张和当前图片大小一样的单通道图片
    img_near_insert = np.zeros([new_h, new_w,c], np.uint8)

    # 计算高和宽的比例
    scale_h =  new_h/h
    scale_w =  new_w/w

    for i in range(new_h):
        for j in range(new_w):
            # 计算在原图上的坐标
            # 这里+0.5是为了四舍五入，int()是向下取整
            x = int(i / scale_h+0.5)
            y = int(j / scale_w+0.5)
            img_near_insert[i, j] = img[x, y]

    return img_near_insert

def near_insert_02(img, new_h, new_w):
    """
    最邻近插值，使用cv2内置函数
    new_h: 新图像的高
    new_w: 新图像的宽
    """
    # 获取图片的high和wide
    h, w ,c= img.shape

    img_near_insert = cv2.resize(img, (new_w,new_h),cv2.INTER_NEAREST)

    return img_near_insert

img_path="206-田海龙-北京/第02周/img/lenna.png"
img=cv_imread(img_path)

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.imshow(img)

img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

img_near_insert=near_insert_01(img, 800, 800)
img_near_insert=cv2.cvtColor(img_near_insert,cv2.COLOR_BGR2RGB)
plt.subplot(222)
plt.imshow(img_near_insert)

img_near_insert=near_insert_02(img, 800, 800)
img_near_insert=cv2.cvtColor(img_near_insert,cv2.COLOR_BGR2RGB)
plt.subplot(223)
plt.imshow(img_near_insert)

plt.show()

