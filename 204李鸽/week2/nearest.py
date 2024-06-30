import numpy as np
import cv2

def nearest_interpolation(img):
    h, w, c = img.shape
    sh = 1332 / h
    sw = 1326 / w
    nearest_img = np.zeros((1332, 1326, c), dtype=img.dtype)
    for i in range(1332):
        for j in range(1326):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            nearest_img[i, j] = img[x, y]
    return nearest_img


img = cv2.imread('Snipaste.png')
zoom = nearest_interpolation(img)
cv2.imshow('original', img)
cv2.imshow('now', zoom)
cv2.waitKey(0)  # 等待用户按键输入，0表示无限等待
cv2.destroyAllWindows()


