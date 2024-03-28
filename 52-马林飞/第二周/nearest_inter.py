import cv2
import numpy as np


# 图片缩小
def imageZoomOut(sImage, dHight, dWeight):
    sh, sw, channels = sImage.shape
    hrate = sh / dHight
    wrate = sw / dWeight
    imageZoomOutEmpty = np.zeros((sh, sw, channels), np.uint8)
    for i in range(dHight):
        for j in range(dWeight):
            imageZoomOutEmpty[i, j] = sImage[int(i * hrate + 0.5), int(j * wrate + 0.5)]
    return imageZoomOutEmpty


# 图片放大
def imageZoom(sImage, dHight, dWeight):
    sh, sw, channels = sImage.shape
    hrate = sh / dHight
    wrate = sw / dWeight
    imageZoomEmpty = np.zeros((dHight, dWeight, channels), np.uint8)
    for i in range(dHight):
        for j in range(dWeight):
            imageZoomEmpty[i, j] = sImage[int(i * hrate + 0.5), int(j * wrate + 0.5)]
    return imageZoomEmpty


if __name__ == '__main__':
    sImage = cv2.imread("123.jpg")

    dImage1 = imageZoomOut(sImage, 300, 300)
    dImage2 = imageZoom(sImage, 800, 800)

    cv2.imshow("org_image", sImage)
    cv2.imshow("zoom_image_out", dImage1)
    cv2.imshow("zoom_image", dImage2)
    cv2.waitKey()
    cv2.destroyAllWindows()
