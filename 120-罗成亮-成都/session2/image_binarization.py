import cv2
import numpy as np


def binarization(img):
    height, width = img.shape[:2]
    new_img = np.zeros([height, width], img.dtype)
    for h in range(height):
        for w in range(width):
            px = img[h, w]
            new_img[h, w] = 0 if int(px[0] * 0.11 + px[1] * 0.59 + px[2] * 0.3) / 255 < 0.5 else 255
    return new_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    binarization_img = binarization(img)

    cv2.imshow("image", img)
    cv2.imshow("image show binarization", binarization_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
