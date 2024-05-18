"""
Canny Threshold
"""
import cv2


def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges,
                               lowThreshold,
                               lowThreshold*ratio,
                               apertureSize=kernel_size)
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    cv2.imshow('canny demo', dst)


if __name__ == '__main__':
    lowThreshold = 0
    max_lowThreshold = 100
    ratio = 3
    kernel_size = 3

    img = cv2.imread('E:/Desktop/jianli/lenna.png', 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建窗口
    cv2.namedWindow('canny demo')
    cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)
    CannyThreshold(0)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
