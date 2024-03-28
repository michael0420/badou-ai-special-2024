import cv2
import numpy as np


def zoom(img, scale):
    height, width, channels = img.shape

    new_height = int(height * scale)
    new_width = int(width * scale)
    new_img = np.zeros([new_height, new_width, channels], np.uint8)
    for h in range(new_height):
        for w in range(new_width):
            x = int(h / scale)
            y = int(w / scale)
            new_img[h, w] = img[x, y]
    return new_img


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    scale = 2
    cv2.imshow("image", img)
    cv2.imshow(f"lennaX{scale}.png", zoom(img, scale))
    cv2.waitKey(0)
