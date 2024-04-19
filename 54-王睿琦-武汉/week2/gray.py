import numpy as np
import cv2
import matplotlib.pyplot as plt


image1 = cv2.imread("liuyifei.jpg") # 读图
image2 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB) #在使用cv2.imread()读取彩色图像后得到的格式是BGR格式，像素值范围在0~255之间，通道格式为(H,W,C)，想要显示RGB类型的图像要进行一步格式转换。
image = cv2.resize(image2,(720,1280)) # 缩放
h, w = image.shape[:2] # 取高度和宽度
imageGray = np.zeros([h, w], image.dtype) # 创建一张和当前图片大小一样的单通道图片

for i in range(h):
    for j in range(w):
        m = image[i, j] # 取出当前high和wide中的BGR坐标
        imageGray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3) # 将BGR坐标转化为gray坐标并赋值给新图像,带公式计算

#方法一展示
# #展示图片
# cv2.imshow("yuantu", image)
# cv2.imshow("gray", imageGray)
# #等待及销毁
# cv2.waitKey()
# cv2.destroyAllWindows()

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(imageGray, cmap='gray')
plt.show()





