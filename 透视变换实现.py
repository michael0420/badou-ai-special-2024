import cv2
import numpy as np

img = cv2.imread("C:/Users/86188/Pictures/photo1.jpg")

result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)


"""
您提供的代码实现了对一张名为 'photo1.jpg' 的图片进行透视变换（perspective transformation）。
代码使用了 OpenCV 的 cv2.getPerspectiveTransform 函数来计算变换矩阵 m，然后使用 cv2.warpPerspective 对图像进行透视变换。
下面是代码的简化和解释：

import cv2 
import numpy as np 
 
# 加载图像 
img = cv2.imread('photo1.jpg')
 
# 定义源和目标顶点坐标 
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
 
# 计算透视变换矩阵 
m = cv2.getPerspectiveTransform(src, dst)
 
# 应用透视变换 
result = cv2.warpPerspective(img, m, (337, 488))
 
# 显示原始图像和变换后的结果 
cv2.imshow('Original Image', img)
cv2.imshow('Warped Image', result)
 
# 等待按键关闭窗口 
cv2.waitKey(0)
cv2.destroyAllWindows()

在这个例子中：

    src 和 dst 分别定义了四个点的坐标，它们分别对应于原始图像和期望经过透视变换后的四个角点。
    cv2.getPerspectiveTransform(src, dst) 生成了一个透视变换矩阵 m。
    cv2.warpPerspective(img, m, (337, 488)) 应用了这个变换矩阵到原始图像 img 上，从而得到变换后的结果。
    最后，原始图像和变换后的图像被显示出来供比较。

请注意，为了保证变换的效果，src 和 dst 中的点应分别对应于图像的四个角点。
而且，变换的结果取决于您提供的顶点坐标，所以请根据实际需要合理设置这些坐标。
"""