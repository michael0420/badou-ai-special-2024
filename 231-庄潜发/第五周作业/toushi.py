"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/5/10 0:20
"""
import cv2
import numpy as np

"""
透视变换需要4个原点,以及4个目标点
通过这8个点进行计算得到透视变换矩阵
"""
src_img = cv2.imread("./photo1.jpg")
# 需要复制一个图像, 这个图像用来变换时使用
temp_img = src_img.copy()
# 四个原点坐标, 以及四个目标点坐标
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
# 生成变换矩阵
m = cv2.getPerspectiveTransform(src, dst)
print(f"变换矩阵:\n{m}")
# 通过变换矩阵计算出目标图像, 第一个参数为要进行变换的图像, 第二个参数为变换矩阵, 第三个表示要输出图像的大小
dst_img = cv2.warpPerspective(temp_img, m, (337, 488))
cv2.imshow("src", src_img)
cv2.imshow("dst", dst_img)
cv2.waitKey()
cv2.destroyAllWindows()
