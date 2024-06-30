import cv2
import numpy as np
###实现K-means
def Kmeans():
    gray_img = cv2.imread('lenna.png', 0)

    h, w = gray_img.shape

    data = gray_img.reshape((h * w, 1))
    data = np.float32(data)

    c = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    flag = cv2.KMEANS_RANDOM_CENTERS

    compactness, labels, centers = cv2.kmeans(data, 4, None, c, 10, flag)

    dst = labels.reshape((h, w))

    K_img = np.zeros([h, w], gray_img.dtype)

    for i in range(h):
        for j in range(w):
            if dst[i][j] == 0:
                K_img[i][j] = 0
            if dst[i][j] == 1:
                K_img[i][j] = 85
            if dst[i][j] == 2:
                K_img[i][j] = 170
            if dst[i][j] == 3:
                K_img[i][j] = 255

    cv2.imshow('kmeans_img', K_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

Kmeans()
