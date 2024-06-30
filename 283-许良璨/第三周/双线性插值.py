import cv2
import numpy as np

def alterimage(h1,w1,img):
    h,w,cannel=img.shape
    if h==h1 and w==w1:
        return img.copy()
    new_img = np.zeros((h1, w1, cannel), dtype = np.uint8)
    sh = h / h1
    sw = w / w1
    for i in range(h1):
        for g in range(w1):
    # 位移
            x = (g + 0.5) * sw - 0.5
            y = (i + 0.5) * sh - 0.5
    # 边界
            x1 = int(x)
            x2 = int(min(w-1, x1 + 1))
            y1 = int(y)
            y2 = int(min(h-1, y1 + 1))

    # 公式
            tamp0 = (x - x1) * img[y1, x2] + (x2 - x) * img[y1, x1]
            tamp1 = (x - x1) * img[y2, x2] + (x2 - x) * img[y2, x1]

            new_img[i, g] = (y - y1) * tamp1 + (y2 - y) * tamp0

    return new_img
if __name__ == '__main__':
    img=cv2.imread("../lenna.png")
    dst = alterimage(800,700,img)
    cv2.imshow("this is the resource!", img)
    cv2.imshow("this is the outcome!",dst)
    cv2.waitKey(0)


