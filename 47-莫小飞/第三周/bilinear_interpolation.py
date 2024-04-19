import numpy as np
import cv2

img = cv2.imread("lenna.png")
src_high, src_width, channel = img.shape

def bilinear(dst_high, dst_width):

    bilinear_img = np.zeros([dst_high, dst_width, channel], img.dtype)
    scale_x, scale_y = float(src_width) / dst_width, float(src_high) / dst_high

    print("x: %s, y: %s" % (scale_x, scale_y))
    print(src_width / dst_width)
    for i in range(channel):
        for h in range(dst_high):
            for w in range(dst_width):

                src_x = (w + 0.5) * scale_x - 0.5
                src_y = (h + 0.5) * scale_y - 0.5

                src_x0 = int(src_x)
                src_x1 = min(src_x0 + 1, src_width - 1)

                src_y0 = int(src_y)
                src_y1 = min(src_y0 + 1, src_high - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]

                result = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

                if h == dst_high - 1 and w == dst_width - 1:
                    print(result)
                bilinear_img[h, w, i] = result

    return bilinear_img


print("start===")
bilinear_img = bilinear(100, 100)
print("end===")
cv2.imshow("bilinear", bilinear_img)
cv2.waitKey(0)

