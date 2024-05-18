import cv2 
import numpy as np

img = cv2.imread(r"E:\imger\photo1.jpg")

img_copy = img.copy()

'''
注意这里src和dst的输入并不是图像。
src:原图中坐标
dst：新图中坐标
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

print(img.shape)
# 生成透视变换矩阵；进行透视变换

t = cv2.getperspectiveTransform(src, dst)

print("warpjz:", t)

trans_img = cv2.warpPerspective(img_copy, t, (337, 448))

cv2.imshow("src", img)
cv2.imshow("trans_img", trans_img)

if cv2.waitKey(0) ==  27:
    cv2.destroyALLWindows()