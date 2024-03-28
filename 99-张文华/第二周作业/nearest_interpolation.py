'''
作业2：
实现最邻近插值法
'''

import cv2
import numpy as np


def near_interp(img):
    h, w, c = img.shape
    img_interp = np.zeros([800, 800, c], img.dtype)
    sh = 800 / h                                        # 取得行列的比例
    sw = 800 / w
    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)                       # 通过比例算的对应的像素点位置
            y = int(j / sh + 0.5)
            img_interp[i, j] = img[x, y]
    return img_interp


if __name__ == '__main__':

    img = cv2.imread('lenna.png')
    imgBig = near_interp(img)
    cv2.imshow('img', imgBig)
    print(imgBig.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
