# # 图片灰度化————方法1。需要安装OpenCV库的环境中运行，pip install opencv-python
# import cv2
#
# img = cv2.imread('lenna.png')                  # 读取照片
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    # 将图片转化成灰度图片,将BGR通道，转换为RGB通道
# cv2.imwrite('gray_lenna.png',gray)             # 保存灰度图像，保存为名为gray_lenna.png的新图片


# # 图片灰度化————方法2。使用python的PIL库（Pillow）实现图片灰度化
# from PIL import Image
#
# img = Image.open('lenna.png')                  # 打开图片
# gray2_img = img.convert('L')                   # 将图片转化成灰度图片
# gray2_img.save('gray2_img.png')                # 保存灰度图像，保存为名为gray2_lenna.png的新图片


# # 图片灰度化————方法3。
import cv2
import numpy as np

img = cv2.imread('lenna.png')                    # 读取照片
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)      # 将图片转化成灰度图片,将BGR通道，转换为RGB通道
print(img.shape)                                 # 图片读取像素行数512，列数512，3通道

h,w = img.shape[:2]                              # 获取图片的high和wide
gray_img = np.zeros((h,w),img.dtype)             # 创建一张和当前图片大小一样的单通道图片
binarization_img = np.zeros((h,w),img.dtype)

# numpy.zeros(shape, dtype=float, order='C')
# 其中，shape参数为一个整数元组，表示数组的形状，如(3, 4)表示3行4列的数组；dtype参数为可选参数，表示数组的数据类型，
# 默认为float类型；order参数为可选参数，表示数组在内存中的存储顺序，默认为'C'（按行存储）。

for i in range(h):
    for j in range(w):
        m = img[i,j]                                              # 取出当前high和wide中的BGR坐标
        gray_img[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)     # 将BGR坐标转化为gray坐标并赋值给新图像

print(m)
print(gray_img)
cv2.imshow('img',gray_img)
key=cv2.waitKey()                                # 窗口持续显示图片

# 灰度转二值化（binarization）
for i in range(h):
    for j in range(w):
        m = gray_img[i,j]
        if m / 255 <= 0.5:
            binarization_img[i,j] = 0            # 该点像素置黑
        else:
            binarization_img[i,j] = 255          # 该点像素置白

cv2.imshow('lenna',img)                          # 图片展示 cv2.imshow(name,img)
cv2.imshow('gray',gray_img)
cv2.imshow('binarization',binarization_img)
cv2.waitKey(0)                                   # 等待时间，表示毫秒级，0代表任意键终止






























