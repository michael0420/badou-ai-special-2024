import cv2
import numpy as np
import matplotlib.pyplot as plt

# 实现彩色图像的灰度化：
## 1.1 原始逻辑实现灰度化：(使用cv2实现)
img = cv2.imread("lenna.png")
height, weight = img.shape[:2]
gray_image = np.zeros((height, weight), dtype=img.dtype)
for i in range(height):
    for j in range(weight):
        m = img[i, j]  # 取出当前点的BGR像数值
        gray = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标像数值转换为灰度值
        gray_image[i, j] = gray  # 将转换后的灰度值赋值给gray_image

# cv2.imshow("Gray Image", gray_image)
# cv2.waitKey(0)   ## 注意使用cv2来实现这两行是必要的，不然图片不会现实出来。 cv2.waitKey是一个用于在窗口上等待指定的毫秒数的函数，它会暂停程序的执行，直到用户按下键盘上的任意键。所以如果不使用cv2.waitKey，窗口会立刻关闭，程序也不会有时间进行显示。
# cv2.destroyAllWindows()

plt.imshow(gray_image, cmap='gray')  # 或者使用plt来展示结果图片。
plt.show()

## 1.2 调用的简单方式是实现灰度化：
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # plt.subplot(221)
# plt.imshow(img_gray, cmap='gray')
# plt.show()


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# 在原始图像处理过程中，我们需要先将其转换为灰度图像，然后再进行二值化处理。
# 实现二值化：(使用cv2来实现)
img2 = cv2.imread("lenna.png")  # 读取原始图片
gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 利用cv2将原始图片转化为灰度图
rows, cols = gray_image.shape[:2]
binary_image = np.zeros((rows, cols), dtype=img2.dtype)
for i in range(rows):  # 再利用灰度图去实现二值化
    for j in range(cols):
        if gray_image[i, j] <= 100:
            binary_image[i, j] = 0
        else:
            binary_image[i, j] = 1
# binary_image = np.where(gray_image >= 0.5, 1, 0)

plt.imshow(binary_image, cmap='gray')
plt.show()  # 使用plt画图，最后写一个plt.show()才会将图片显示出来


