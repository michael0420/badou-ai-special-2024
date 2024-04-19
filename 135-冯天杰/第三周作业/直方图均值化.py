import cv2
from matplotlib import pyplot as plt
# 解决subplot中文乱码
from matplotlib.font_manager import FontProperties

font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=8)

# 调整子图像位置
fig, ax = plt.subplots(4, 2)
fig.tight_layout()

# 原图灰度图
plt.subplot(421)
img = plt.imread("69e754ae5b5a3ecf8f150894ee6b095.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.title("原图灰度图", fontproperties=font_set)
plt.imshow(gray, cmap='gray')

# 原图灰度图直方图
plt.subplot(422)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.hist(gray.ravel(), 256)
plt.title("原图灰度图直方图", fontproperties=font_set)

# 原图灰度图均值化
plt.subplot(423)
gray_equal = cv2.equalizeHist(gray)
plt.imshow(gray_equal, cmap="gray")
plt.title("原图灰度图均值化", fontproperties=font_set)

# 原图灰度图均值化直方
plt.subplot(424)
hist_gray_equal = cv2.calcHist([gray_equal], [0], None, [256], [0, 256])
plt.hist(gray_equal.ravel(), 256)
plt.title("原图灰度图均值化直方图", fontproperties=font_set)

# 原图
plt.subplot(425)
plt.title("原图", fontproperties=font_set)
plt.imshow(img)

# 原图直方图
plt.subplot(426)
color = ("b", "g", "r")
for i, color_equal in enumerate(color):
    hist_img = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.title("原图直方图", fontproperties=font_set)
    plt.plot(hist_img, color=color_equal)

# 原图均值化
plt.subplot(427)
b, g, r = cv2.split(img)
hist_r = cv2.equalizeHist(r)
hist_g = cv2.equalizeHist(g)
hist_b = cv2.equalizeHist(b)
img_equal = cv2.merge((hist_b, hist_g, hist_r))
plt.title("原图均值化", fontproperties=font_set)
plt.imshow(img_equal)

# 原图均值化直方图
plt.subplot(428)
for i, color in enumerate(color):
    hist_img_equal = cv2.calcHist([img_equal], [i], None, [256], [0, 256])
    plt.title("原图均值化直方图", fontproperties=font_set)
    plt.plot(hist_img_equal, color=color)

plt.show()
