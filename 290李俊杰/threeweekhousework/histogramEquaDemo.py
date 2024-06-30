# 【第三周作业】
#
#  2.实现直方图均衡化

import cv2
import numpy as np
from matplotlib import pyplot as plt
# 用cv2.imread读的图，通道顺序是bgr，用plt.imshow显示得转成rgb
img=cv2.imread("lenna.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.imshow(img)
# 实现灰度图直方图均衡化
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gh,gw=img_gray.shape
print(gh,gw)
# 计算灰度直方图的累加直方图和直方图均衡化已经有封装好的函数，直接使用即可
dst = cv2.equalizeHist(img_gray)
newdst=cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
plt.subplot(222)
plt.imshow(newdst)


# 实现彩色图直方图均衡化
img=cv2.imread("lenna.png")
# 读取原图的高宽通道
# h,w,c=img.shape
# # 生成一个跟原图一样高宽通道的空图片
# dstimg=np.zeros((h,w,c),dtype=np.uint8)
# 对图片的每个通道做直方图均衡化
# 首先分解每个通道cv2.split
(b, g, r) = cv2.split(img)
dstimgb=cv2.equalizeHist(b)
dstimgg=cv2.equalizeHist(g)
dstimgr=cv2.equalizeHist(r)
# 在进行合并操作cv2.merge
dstimg = cv2.merge((dstimgb, dstimgg, dstimgr))
newdstimg =cv2.cvtColor(dstimg,cv2.COLOR_BGR2RGB)
plt.subplot(223)
plt.imshow(newdstimg)
plt.show()

