import  cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('lenna.png')

rows,lines,channels=img.shape

img_insert= np.zeros((800,800,channels),dtype=img.dtype)

#计算放大长宽比例
sh = 800 / rows
sw = 800 / lines
for i in range(800):
    for j in range(800):
        # 加 1/2 是因为 int是向下取整，会把小数点后面直接截断， 1.6+ 1/2 会让图像的坐标值本来就大的一方偏向更大的那一方，
        # 更小的一方也不会超过太多 例如： 1.3 + 1/2 = 1.5  int（1.5）=1
        # 相当于四舍五入
        x = int(i / sh + 0.5)
        y = int(j / sh + 0.5)
        img_insert[i,j]=img[x,y]

print(img_insert.shape)

cv2.imshow("orginPicture: ",img)
cv2.imshow("NearInsert:",img_insert)


cv2.waitKey(0)






