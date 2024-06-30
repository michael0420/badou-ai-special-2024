import cv2
import numpy as np

img = cv2.imread('photo1.jpg')
print(img.shape)

'''
    这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
        src和dst中的x,y含义：
        x：表示源图像中点的水平坐标。在像素坐标系中，这通常是相对于图像左上角的横向距离。
        y：表示源图像中点的垂直坐标。同样，在像素坐标系中，这是相对于图像左上角的纵向距离。
'''
src = np.float32([[275, 275], [12, 565], [578, 357], [448, 791]])
dst = np.float32([[0, 0], [0, 550], [400, 0], [400, 550]])

# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)   #这里src中的坐标值顺序为(w,h) 或是(col,row)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(img, m, (400, 550))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
