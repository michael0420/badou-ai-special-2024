import cv2
import numpy as np

def perspective_transform(image, src_points, dst_points):
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 应用透视变换
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    
    return warped

if __name__ == "__main__":
    # 读取图像
    image = cv2.imread('test.png')

    # 定义源点和目标点
    src_points = np.float32([[150, 100], [400, 100], [80, 400], [480, 400]])
    dst_points = np.float32([[0, 0], [300, 0], [0, 400], [300, 400]])

    # 应用透视变换
    warped_image = perspective_transform(image, src_points, dst_points)

    # 显示结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Warped Image', warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
