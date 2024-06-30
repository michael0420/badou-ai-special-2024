# -*- coding: utf-8 -*-
"""
@author: zhjd

"""
import cv2
import numpy as np


def perspective_transform(src_points, dst_points, image_path):
    img = cv2.imread(image_path)
    result3 = img.copy()

    m = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(result3, m, (337, 488))

    cv2.imshow("Original Image", img)
    cv2.imshow("Transformed Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    src_points = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst_points = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

    image_path = 'photo1.jpg'

    perspective_transform(src_points, dst_points, image_path)
