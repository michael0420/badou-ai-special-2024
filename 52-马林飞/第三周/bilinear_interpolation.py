import cv2
import numpy as np

img_src = cv2.imread("lenna.png")
sh, sw, channels = img_src.shape
print(sh, sw, channels)

dh = 800
dw = 800

rh = sh / dh
rw = sw / dw

print(rh, rw)

img_dest = np.zeros((dh, dw, channels), np.uint8)

for x in range(channels):
    for i in range(dh):
        for j in range(dw):
            src_x = (i + 0.5) * rh - 0.5
            src_y = (j + 0.5) * rw - 0.5

            src_x0 = int(np.floor(src_x))
            src_x1 = min(src_x0 + 1, sh - 1)
            src_y0 = int(np.floor(src_y))
            src_y1 = min(src_y0 + 1, sw - 1)

            temp0 = (src_x1 - src_x) * img_src[src_x0, src_y0, x] + (src_x - src_x0) * img_src[src_x0, src_y1, x]
            temp1 = (src_x1 - src_x) * img_src[src_x1, src_y0, x] + (src_x - src_x0) * img_src[src_x1, src_y1, x]
            img_dest[i, j, x] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

cv2.imshow("img_src", img_src)
cv2.imshow("img_desc", img_dest)
cv2.imshow("img_resize", cv2.resize(img_dest, (dh,dw)))
cv2.waitKey(0)


