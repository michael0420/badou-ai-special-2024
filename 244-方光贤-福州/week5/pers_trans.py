import cv2
import numpy as np

'''
算法逻辑
透视变换：将曲线变直线 矫正图像 
根据公式 a11x+a12y+a13-a31xX'-a32X'y = X'
        a21x+a22y+a23-a31xY'-a32yY' = Y'
带入源点四个坐标和目标点四个坐标解方程
'''
img = cv2.imread("photo1.jpg")
#复制一个原始图像
result0 = img.copy()
print(img.shape)
#原始图像和目标图像四个顶点的坐标
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

#带入透视变换矩阵并打印转换矩阵
warp_matrix = cv2.getPerspectiveTransform(src, dst)
print("warp_matrix : ")
print(warp_matrix)
#生成透视变换后的图像 调用函数 参数分别为原图 转移矩阵 转移后的图像大小
result1 = cv2.warpPerspective(result0, warp_matrix, (337,488))
cv2.imshow("src", img)
cv2.imshow("result", result1)
cv2.waitKey(0)
cv2.destroyAllWindows()

