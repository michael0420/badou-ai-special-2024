"""
@author: 74+张永刚
彩色图像的灰度化
"""

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 读取图片
img = cv2.imread("img/lenna.png")
# 获取图片的hide、wide
hide,wide = img.shape[:2]
# 创建一张和当前图片一样大小的单通道图片
img_gray = np.zeros([hide,wide],img.dtype)
"""
1、浮点算法：Gray=R(0.3)+G(0.59)+B(0.11)
2、整数算法：Gray=(R(30)+G(59)+B(11))/100
"""

# 将图片灰度化
for i in range(hide):
    for j in range(wide):
        # 取出当前 hide和wide中的BRG坐标
        m = img[i,j]
        # 将BRG坐标转换为gray坐标并赋值给新图片对应的坐标位置
        # 这里需要注意cv2获取到的图片为BGR
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
print(m)
print(img_gray)
print("image show gray: %s" % img_gray)
cv2.imshow("image show gray",img_gray) # 这里跟老师的代码走 没有显示出图片，是因为后面需要加cv2.waitKey(0) 才会显示出来
# 生成图片后 将持续显示，直到有键盘的按键被按下
cv2.waitKey(0)
# 关闭所有打开的图片窗口，释放系统资源
cv2.destroyAllWindows()
