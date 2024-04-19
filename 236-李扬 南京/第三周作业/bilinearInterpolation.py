import cv2 as cv
import numpy as np

#最邻近插值
def bilinearInterpolationfun(img, newsize):
    h, w, channels = img.shape[:]
    newh, neww = newsize[0], newsize[1]
    newImg = np.zeros((newh, neww, channels), dtype=np.uint8)
    if h == newh and w == neww:
        return  img.copy()
    ratioH = float(newh) / h        #获取高度比例
    ratioW = float(neww) / w        #获取宽度比例
    for i in range(channels):
         for y in range(newh):
             for x in range(neww):
                srcx0 = (x + 0.5) / ratioW - 0.5
                srcy0 = (y + 0.5) / ratioH - 0.5

                #获取对应原图中的邻近四点
                y0 = int(srcy0)
                y1 = min(y0 + 1, h - 1)
                x0 = int(srcx0)
                x1 = min(x0 + 1, w - 1)

                #分解公式，第一步获取f(R1)和f(R2)
                fr1 = (x1 - srcx0) * img[y0, x0, i] + (srcx0 - x0) * img[y0, x1, i]
                fr2 = (x1 - srcx0) * img[y1, x0, i] + (srcx0 - x0) * img[y1, x1, i]

                #根据f(R1)和f(R2)计算出目标值
                newImg[y, x, i] = int((y1 - srcy0) * fr1 + (srcy0 - y0) * fr2)

    return newImg
#最邻近插值
img  = cv.imread("lenna.png")
newImg = bilinearInterpolationfun(img, [700, 700])
cv.imshow("test", newImg)
cv.waitKey()