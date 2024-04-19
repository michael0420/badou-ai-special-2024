import cv2
import numpy as np

def function(img,newHeight,newWidth):
    height, width, channels = img.shape
    emptyImage = np.zeros((newHeight, newWidth, channels), np.uint8)
    sh = newHeight / height
    sw = newWidth / width
    for i in range(newHeight):
        for j in range(newWidth):
            x = int(i / sh + 0.5)  # int(),转为整型，使用向下取整
            y = int(j / sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage

img = cv2.imread("lenna.png")
newimg = function(img,800,800)
print(newimg)
print(newimg.shape)
cv2.imshow("nearest interp", newimg)
cv2.imshow("image", img)
cv2.waitKey(0)


