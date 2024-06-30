import cv2 as cv
import numpy as np

if __name__ == '__main__':
    img = cv.imread('../photo1.jpg')
    img2 = img.copy()

    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    # cv.getPerspectiveTransform函数期望源点(src)和目标点(dst)这些点都应该是浮点数(CV_32F类型)
    m = cv.getPerspectiveTransform(src, dst)
    result = cv.warpPerspective(img2, m, (337, 488))
    cv.imshow("1", img)
    cv.imshow("2", result)
    cv.waitKey(0)
    cv.destroyAllWindows()
