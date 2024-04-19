import numpy as np
import cv2

def bilinear_interpolation(img,dstSize):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = dstSize
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros([dst_h, dst_w, channel],dtype=np.uint8)
    x_scale, y_scale = float(src_w / dst_w), float(src_h / dst_h)
    for c in range(channel):
        for h in range(dst_h):
            for w in range(dst_w):
                x = (w + 0.5) * x_scale - 0.5
                y = (h + 0.5) * y_scale - 0.5

                x0 = int(np.floor(x))
                x1 = min(x0 + 1, src_w - 1)
                y0 = int(np.floor(y))
                y1 = min(y0 + 1, src_h - 1)

                temp_0 = (x1 - x) * img[y0, x0, c] + (x - x0) * img[y1, x0, c]
                temp_1 = (x1 - x) * img[y1, x0, c] + (x - x0) * img[y1, x1, c]
                dst_img[h, w, c] = int((y1 - y) * temp_0  + (y - y0) * temp_1)
    return dst_img


if __name__ == "__main__":
    img = cv2.imread('lenna.png')
    dstSize = (800,800)
    dst_img = bilinear_interpolation(img,dstSize)
    cv2.imshow("bilinear interpolation",dst_img)
    cv2.waitKey(0)