"""
手动实现椒盐噪声
"""
import random
import cv2

perpectage = 0.1  # 设置椒盐噪声的百分比


def PepperSaltNoise(src, channel=1):
    newImg = src.copy()
    for c in range(channel):
        newimg_c=newImg[:,:,c]
        newImgNum = int(newimg_c.shape[0] * newimg_c.shape[1] * perpectage)
        for i in range(newImgNum):
            x = random.randint(0, newimg_c.shape[0] - 1)
            y = random.randint(0, newimg_c.shape[1] - 1)
            rad = random.randint(0, 1)
            if rad == 0:
                newImg[x, y,c] = 0
            else:
                newImg[x, y,c] = 255
    return newImg


src = cv2.imread("../lenna.png")
_,_,channel=src.shape
newimg = PepperSaltNoise(src,channel)
# src2=cv2.imread("../lenna.png")
# src_gray=cv2.cvtColor(src2,cv2.COLOR_BGR2GRAY)
cv2.imshow("src", src)
cv2.imshow("newimg", newimg)
cv2.waitKey(0)
