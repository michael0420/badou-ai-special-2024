import cv2
import numpy as np
def function(img):
    height,width,channels = img.shape
    # 将img扩展为800 * 800像素点的图片
    emptyImage = np.zeros((800,800,channels),np.uint8)
    sh = 800 / height # 等比例的高和宽
    sw = 800 / width
    for i in range(800):
        for j in range(800):
            # x,y为img的坐标
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            emptyImage[i,j] = img[x,y]
    return emptyImage

img = cv2.imread("lenna.png")
zoom = function(img)
cv2.imwrite("lenna_large.png",zoom)