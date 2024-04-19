import numpy as np
import cv2


# 双线性插值
def bili_near_interpolation(img, num_h, num_w):
    original_h, original_w, original_t = img.shape
    wtq_img_r = np.zeros([num_h, num_w, original_t], np.uint8)

    for k in range(original_t):
        for j in range(num_h):
            for i in range(num_w):
                x = (i+0.5)*original_w / num_w - 0.5
                y = (j+0.5)*original_h / num_h - 0.5
                x0 = int(np.floor(x))
                x1 = min(x0+1, original_w-1)
                y0 = int(np.floor(y))
                y1 = min(y0+1, original_h-1)

                z0 = (x1-x) * wtq_img[y0, x0, k] + (x-x0) * wtq_img[y0, x1, k]
                z1 = (x1-x) * wtq_img[y1, x1, k] + (x-x0) * wtq_img[y1, x1, k]
                wtq_img_r[j, i, k] = int((y1-y)*z0+(y-y0)*z1)
    return wtq_img_r


wtq_img = cv2.imread('lenna.png')
wtq_img_ni = bili_near_interpolation(wtq_img, 800, 800)
cv2.imshow("lenna", wtq_img)
cv2.imshow("lenna_nl", wtq_img_ni)
cv2.waitKey(0)