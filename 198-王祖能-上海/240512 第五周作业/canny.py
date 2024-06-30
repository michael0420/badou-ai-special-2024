# canny边缘检测接口

import cv2
import numpy as np
'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])
threshold1：阈值1（最小值） 阈值越小，细节边缘越多
threshold2：阈值2（最大值）
apertureSize：sobel算子（卷积核）大小，为>3奇数，越大的细节边缘越多
L2gradient ：布尔值。
True：使用更精确的L2范数进行计算（即两个方向的导数的平方和再开方）L2gradient=True时，检测出的边缘减少了
False：使用L1范数（直接将两个方向导数的绝对值相加）
'''

img = cv2.imread('lenna.png', 0)  # 灰度化图像效率更高，也更准确
img_edge1 = cv2.Canny(img, 100, 200)  # threshold阈值1,2
img_edge2 = cv2.Canny(img, 150, 250)
img_edge3 = cv2.Canny(img, 150, 250, L2gradient=True)  # L2范数
img_edge4 = cv2.Canny(img, 150, 250, apertureSize=5)  # Sobel算子核大小，建议3
cv2.imshow('1', np.hstack([img_edge1, img_edge2, img_edge3, img_edge4]))
cv2.waitKey()
cv2.destroyAllWindows()

# 滞后阈值 minVal 和 maxVal。强度梯度大于 maxVal 肯定是边缘，小于 minVal 肯定是非边缘被丢弃
# 右边图像呈现的信息少，因为低阈值高，潜在边缘在低阈值以下被舍弃，而且有的边缘信息达不到高阈值也被舍弃。
# 针对不同的图片阈值的范围需要不断的调式有最佳的效果。
