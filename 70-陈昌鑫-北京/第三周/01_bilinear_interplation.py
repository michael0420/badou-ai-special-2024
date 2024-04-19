import numpy as np
import cv2

def bilinear_interplation(src_img, des_img):
    src_h, src_w = src_img.shape[0], src_img.shape[1]
    des_h, des_w = des_img.shape[0], des_img.shape[1]
    if src_h == des_h and src_w == des_w:
        return src_img.copy()
    for i in range(3):
        for des_y in range(des_h):
            for des_x in range(des_w):
                src_x = (des_x + 0.5) * src_w / des_w - 0.5
                src_y = (des_y + 0.5) * src_h / des_h - 0.5

                src_x1 = int(np.floor(src_x))
                src_x2 = min(src_x1 + 1, src_w - 1)
                src_y1 = int(np.floor(src_y))
                src_y2 = min(src_y1 + 1, src_h - 1)

                tmp1 = (src_x2 - src_x) * src_img[src_y1, src_x1, i] + (src_x - src_x1) * src_img[src_y1, src_x2, i]
                tmp2 = (src_x2 - src_x) * src_img[src_y2, src_x1, i] + (src_x - src_x1) * src_img[src_y2, src_x2, i]
                des_img[des_y, des_x, i] = (src_y2 - src_y) * tmp1 + (src_y - src_y1) * tmp2
    return des_img

src_img = cv2.imread("lenna.png")
des_img = np.zeros((770,660,3), np.uint8)
des_img = bilinear_interplation(src_img, des_img)
cv2.imshow("src_img", src_img)
cv2.imshow("des_img", des_img)
cv2.waitKey(0)