import cv2
import numpy as np

img = cv2.imread("photo1.jpg")

result = img.copy()
#getPerspectiveTransform 函数期望的输入是 4 个两维浮点型坐标点  dtype=np.float32
#4个顶点
vertex = np.array([[207, 151], [517, 285], [17, 601], [343, 731]],dtype=np.float32)
#4个目标点
destination = np.array([[0,0],[0,512],[1024,0],[1024,512]],dtype=np.float32)
print(img.shape)

#生成透视变换矩阵(3x3)
m = cv2.getPerspectiveTransform(vertex,destination)
#cv2.warpPerspective执行透视变换
#输入图像、变换矩阵以及输出图像的尺寸作为参数
result_final = cv2.warpPerspective(result,m,(1024,512))
cv2.imshow("original",img)
cv2.imshow("result",result_final)
cv2.waitKey(0)