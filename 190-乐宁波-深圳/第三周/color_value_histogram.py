import cv2
import matplotlib.pyplot as plt

gray_img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)  # 使用cv2.IMREAD_GRAYSCALE标志将其作为灰度图像加载

'''
[gray_img]是包含要分析的图像的列表，
[0]指定我们只对第一个（也是唯一的）通道感兴趣，即灰度通道。
None表示我们没有使用掩码，
[256]定义了直方图的bin数量，即灰度级数。
[0, 256]定义了直方图的范围，从0到255
'''
hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
# 创建一个图形窗口
plt.figure(figsize=(10, 5))

plt.plot(hist, color='gray', linewidth=2, label='gray')  # hist是直方图数据，color='blue'设置线条颜色为蓝色，linewidth=2设置线条宽度为2个像素。

# 读取彩色图像
color_img = cv2.imread('img.jpg', cv2.IMREAD_COLOR)

# 计算每个通道的直方图
histBlue = cv2.calcHist([color_img], [0], None, [256], [0, 256])
histGreen = cv2.calcHist([color_img], [1], None, [256], [0, 256])
histRed = cv2.calcHist([color_img], [2], None, [256], [0, 256])

plt.plot(histBlue, color='blue', linewidth=2, label='blue')
plt.plot(histGreen, color='green', linewidth=2, label='green')
plt.plot(histRed, color='red', linewidth=2, label='red')

# 设置标题和轴标签
plt.title('RGB Color Histogram')
plt.xlabel('Intensity')
plt.ylabel('Number of Pixels')
plt.xlim([0, 256])  # 设置x轴的显示范围，从0到256
# 添加图例
plt.legend()
# 显示图形
plt.show()
