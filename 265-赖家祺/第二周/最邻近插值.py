import cv2
import numpy as np

"""
最邻近插值 
"""

# 放大
def zoomIn(img):
    height, width, chans = img.shape
    emptyImage = np.zeros([768, 768, chans], np.uint8)
    sh = height / 768
    sw = width / 768

    for h in range(768):
        for w in range(768):
            x = round(h * sh)
            y = round(w * sw)
            emptyImage[h, w] = img[x, y]

    return emptyImage


# 缩小
def zoomOut(img):
    height, width, chans = img.shape
    new_height, new_width = 200, 200
    emptyImage = np.zeros([new_height, new_width, chans], np.uint8)
    sh = new_height / height
    sw = new_width / width

    for h in range(new_height):
        for w in range(new_width):
            x = int(h / sh + 0.5)
            y = int(w / sw + 0.5)
            emptyImage[h, w] = img[x, y]

    return emptyImage


img = cv2.imread("./lenna.png")
cv2.imshow("origin", img)

big = zoomIn(img)
cv2.imshow("big img", big)

small = zoomOut(img)
cv2.imshow("small img", small)

cv2.waitKey(0)
