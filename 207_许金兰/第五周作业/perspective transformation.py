"""
@author: 207-xujinlan
cv2实现透视变换
"""

import cv2
import numpy as np

# 1.读入图片
pic_path = 'photo1.jpg'
img = cv2.imread(pic_path)
# 2.设置原图片中四个点对应新图片中的四个点，这八个点的坐标，需要手动确定
src_vertex = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst_vertex = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
# 3.生成透视变换矩阵
m = cv2.getPerspectiveTransform(src_vertex, dst_vertex)
# 4.进行透视变换
img_perspect = cv2.warpPerspective(img, m, (337, 488))
# 5.图片展示
cv2.imshow("source img", img)
cv2.imshow("Perspective img", img_perspect)
cv2.waitKey(0)
