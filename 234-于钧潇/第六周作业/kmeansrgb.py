import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("catdog.jpg", 0)

    #  转成1维
    data = img.reshape((img.shape[0]*img.shape[1], 1))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 10, 1.0)
    retval, bestLabels, centers = cv2.kmeans(data, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    print(centers)
    centers = np.uint8(centers)
    dst = centers[bestLabels.flatten()]
    dst = dst.reshape((img.shape[0], img.shape[1]))
    cv2.imshow("src", img)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)