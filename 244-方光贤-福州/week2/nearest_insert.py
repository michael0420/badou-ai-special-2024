import cv2
import numpy as np

def insert(img):
    #读取原有图像属性
    height, width, channel = img.shape
    #创建空图像为800*800三通道
    null_image = np.zeros((800,800,channel),np.uint8)
    #比例关系
    sample_height = 800/height
    sample_width = 800/width
    #遍历rgb坐标
    for i in range(800):
        for j in range(800):
            x=int(i/sample_height + 0.5)  #向下取整 手动四舍五入
            y=int(j/sample_width + 0.5)
            null_image[i,j]=img[x,y]
    return null_image

img = cv2.imread("lenna.png")  #读图
new_image = insert(img)  #调用函数
cv2.imshow("image",img)
cv2.imshow("nearest interp",new_image)
cv2.waitKey(0)