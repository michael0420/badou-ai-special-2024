
import cv2
import numpy as np
import matplotlib.pyplot as plt

from Util import cv_imread

def zoom_img_01(img, new_h, new_w):
    """
    缩放图像，自己写的方法
    new_h: 新图像的高
    new_w: 新图像的宽
    """
    # 获取图片的high和wide
    h, w ,c= img.shape

    img_zoom = np.zeros([new_h, new_w,c], np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            # 计算在原图上的坐标
            # 这里+0.5是为了四舍五入，int()是向下取整
            x = int(i / new_h * h+0.5)
            y = int(j / new_w * w+0.5)
            img_zoom[i, j] = img[x, y]

    return img_zoom

def zoom_img_02(img, new_h, new_w):
    """
    缩放图像，使用cv2内置函数
    new_h: 新图像的高
    new_w: 新图像的宽
    """
    # 获取图片的high和wide
    h, w ,c= img.shape

    img_zoom = cv2.resize(img, (new_w,new_h),cv2.INTER_LINEAR)

    return img_zoom

img_path="206-田海龙-北京/第02周/img/lenna.png"
img=cv_imread(img_path)

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.imshow(img)

img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

img_zoom=zoom_img_01(img, 1000, 1000)
img_zoom=cv2.cvtColor(img_zoom,cv2.COLOR_BGR2RGB)
plt.subplot(222)
plt.imshow(img_zoom)

img_zoom=zoom_img_02(img, 50, 50)
img_zoom=cv2.cvtColor(img_zoom,cv2.COLOR_BGR2RGB)
plt.subplot(223)
plt.imshow(img_zoom)

img_zoom=zoom_img_02(img, 100, 100)
img_zoom=cv2.cvtColor(img_zoom,cv2.COLOR_BGR2RGB)
plt.subplot(224)
plt.imshow(img_zoom)

plt.show()
