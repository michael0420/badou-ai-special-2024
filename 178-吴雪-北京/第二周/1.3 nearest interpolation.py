import cv2
import numpy as np
def nearest_interpolation(img, out_dst):
    src_h, src_w, channels = img.shape
    dst_h, dst_w = out_dst[0], out_dst[1]
    virtualPixels = np.zeros([dst_h, dst_w, channels], dtype=np.uint8)
    sh = dst_h / src_h
    sw = dst_w / src_w
    for i in range(dst_h):
        for j in range(dst_w):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            virtualPixels[i, j] = img[x, y]
    return virtualPixels


if __name__ == '__main__':
    img = cv2.imread("E:/Desktop/jianli/lenna.png")
    dst = nearest_interpolation(img, (800, 800))
    print(dst)
    print(dst.shape)
    cv2.imshow("the nearest interpolation", dst)
    cv2.imshow("image show", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
