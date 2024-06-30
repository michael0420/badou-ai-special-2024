'''
【第四周作业】

作业：1.实现高斯噪声 2.实现椒盐噪声 3.实现PCA  4.拓展：证明中心化协方差矩阵公式

'''

# 2.实现椒盐噪声
'''
给一副数字图像加上椒盐噪声的处理顺序：
1.指定信噪比 SNR（信号和噪声所占比例） ，其取值范围在[0, 1]之间
2.计算总像素数目 SP， 得到要加噪的像素数目 NP = SP * SNR
3.随机获取要加噪的每个像素位置P（i, j）
4.指定像素值为255或者0。
5.重复3, 4两个步骤完成所有NP个像素的加噪
'''

import random
import cv2
from skimage import util
# 以灰度图方式读图
img = cv2.imread("lenna.png", 0)
# cv2.imshow("img show",img)
# 指定信噪比 SNR
snr = 0.4
# 计算总像素数目 SP
h, w = img.shape
sp = h * w
# 得到要加噪的像素数目 NP = SP * SNR
np = int(sp * snr + 0.5)
# 随机获取要加噪的每个像素位置P（i, j）
# 指定像素值为255或者0。
# random.random()生成0-1之间的浮点数
for i in range(np):
    randomX = random.randint(0, h - 1)# 不对边缘进行加噪
    randomY = random.randint(0, w - 1)# 不对边缘进行加噪
    if random.random()<=0.5:
        img[randomX,randomY]=255
    elif random.random()<1:
        img[randomX, randomY] = 0

cv2.imshow("saltpper img show",img)
cv2.waitKey(0)

# #可以使用封装的函数实现椒盐噪声util.random_noise(img,mode='s&p')
# # amount： 可选的，float型，是椒盐噪声所占比例，默认值=0.05
# salt_vs_pepper：可选的，float型，椒盐噪声中椒盐比例，值越大表示盐噪声越多，默认值=0.5，即椒盐等量
# img1 = cv2.imread("lenna.png", 0)
# img1=util.random_noise(img,mode='s&p',amount=0.01)
# cv2.imshow("saltpper img show",img1)
# cv2.waitKey(0)