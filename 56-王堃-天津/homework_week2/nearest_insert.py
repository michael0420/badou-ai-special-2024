import cv2
import numpy as np


def amplify(image, new_height, new_width):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    result_image = np.zeros((new_height, new_width, channels), np.uint8)
    scale_height = new_height / height
    scale_width = new_width / width
    for h in range(new_height):
        for w in range(new_width):
            original_h = int(h / scale_height + 0.5)
            original_w = int(w / scale_width + 0.5)
            result_image[h, w] = image[original_h, original_w]
    return result_image


if __name__ == '__main__':
    image = cv2.imread("lenna.png")
    cv2.imshow("original image", image)
    cv2.imshow("amplified image", amplify(image, 800, 800))
    cv2.waitKey(0)
