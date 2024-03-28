import matplotlib.pyplot as plt
import cv2

# 灰度化 整数方式转
# img = cv2.imread("lenna.png")
# img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# high,wide = img_rgb.shape[:2]
# img_gray = np.zeros([high,wide],img_rgb.dtype)
# for i in range(high):
#     for j in range(wide):
#         m = img[i,j]
#         img_gray[i,j] = int((m[0]*30 + m[1]*59 + m[2]*11)/100)   #将RGB坐标转化为gray坐标并赋值给新图像
# print("image show gray: %s"%img_gray)
# cv2.imshow("image show gray",img_gray)
# cv2.waitKey()
# cv2.destroyAllWindows()

# # 灰度化  浮点公式转
# img = cv2.imread("lenna.png")
# img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# high,weight = img_rgb.shape[:2]
# img_gray = np.zeros([high,weight],img_rgb.dtype)
# for i in range(high):
#     for j in range(weight):
#         m = img[i,j]
#         img_gray[i,j] = int(m[0]*0.3 + m[1]*0.59 + m[2]*0.11)   #将RGB坐标转化为gray坐标并赋值给新图像
# print("image show gray: %s"%img_gray)
# cv2.imshow("image show gray",img_gray)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 灰度化 函数转
# img = cv2.imread("lenna.png")
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("lenna gray",img_gray)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 二值化
img = cv2.imread("lenna.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.imshow(img)
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')
# 全局阈值二值化
retval, img_binary1 = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
plt.subplot(223)
plt.imshow(img_binary1,cmap='gray')
# 自适应阈值二值化
img_binary2 = cv2.adaptiveThreshold(
    img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 127, 1)
plt.subplot(224)
plt.imshow(img_binary2,cmap='gray')
plt.show()
# 显示图像
# cv2.imshow('original', img_gray)
# cv2.imshow('binary', img_binary1)
# cv2.waitKey()
# cv2.destroyAllWindows()

