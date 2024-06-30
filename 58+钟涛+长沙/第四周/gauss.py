#高斯噪声

import numpy as np
import random
import cv2
from skimage import util

def gauss(img,mu,segma,p):
    #获取图片长和宽
    h,w = img.shape
    #获取噪声的百分比
    size = int(p*h*w)
    #创建一img大小的空白数组
    desc = np.zeros((h,w), np.uint8)

    for i in range(size):
        #随机位置
        x = random.randint(0, h-1)
        y = random.randint(0,w - 1)
        #随机点 加上高斯噪声
        desc[x,y] = img[x,y] + random.gauss(mu,segma)
    return desc

#读取灰度图片
img = cv2.imread("lenna.png",0)
desc = gauss(img,2,4,0.01)

#图片合并
gauss_img = cv2.addWeighted(img, 0.5, desc, 0.5,0)

#合并展示
cv2.imshow("图片", np.hstack((img,gauss_img)))
cv2.waitKey(0)
cv2.destroyAllWindows()


#api实现
img = cv2.imread("lenna.png",0)
gauss_img = util.random_noise(img, mode="s&p")
cv2.imshow("图片", gauss_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
