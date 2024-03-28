import cv2
import numpy as np


#读取图片
org_img = cv2.imread("../../lenna.png")

#初始化灰度图
h,w = org_img.shape[:2]
gray_img = np.zeros((h,w),dtype=np.uint8)

for i in range(h):
    for j in range(w):
        gray_img[i,j] = org_img[i,j][0]*0.11 + org_img[i,j][1]*0.59 + org_img[i,j][2]*0.3


#对灰度图进行二值化
binary_img = gray_img / 255
binary_img = np.where(binary_img >= 0.5,1,0)
#cv2只能识别uint8类型，所以需要转换为uint8
#前一步转换为0和1后，像素值太暗，导致图片显示后全黑，所以乘255增加亮度
binary_img= (binary_img*255).astype(np.uint8)

cv2.imshow("binary_img",binary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()








