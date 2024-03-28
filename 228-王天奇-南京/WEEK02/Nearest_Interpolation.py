import numpy as np
import cv2


# 最临近插值
def nearest_interpolation(img, num_h, num_w):
    h, w, t = img.shape
    h1 = h / num_h
    w1 = w / num_w
    wtq_img_nl = np.zeros([num_h, num_w, t], np.uint8)

    for i in range(num_h):
        for j in range(num_w):
            h2 = int(h1 * i)
            w2 = int(w1 * j)
            wtq_img_nl[i, j] = wtq_img[h2, w2]
    return wtq_img_nl


wtq_img = cv2.imread('lenna.png')
wtq_img_nl = nearest_interpolation(wtq_img, 200, 200)

cv2.imshow("lenna", wtq_img)
cv2.imshow("lenna_nl", wtq_img_nl)
cv2.waitKey(0)
