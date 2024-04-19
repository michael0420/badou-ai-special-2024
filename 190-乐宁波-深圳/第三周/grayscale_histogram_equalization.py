import cv2
import matplotlib.pyplot as plt

gray_img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)  # 使用cv2.IMREAD_GRAYSCALE标志将其作为灰度图像加载

equilibrated_image = cv2.equalizeHist(gray_img)

plt.figure(figsize=(10, 5))

plt.subplot(221)
plt.imshow(gray_img, cmap='gray')
plt.title('Original Image')

plt.subplot(222)
plt.imshow(equilibrated_image, cmap='gray')
plt.title('Equalized Image')

plt.subplot(223)
hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
plt.plot(hist, color='gray', linewidth=2, label='gray')  # hist是直方图数据，color='blue'设置线条颜色为蓝色，linewidth=2设置线条宽度为2个像素。
plt.ylabel('Number of Pixels')
plt.xlim([0, 256])  # 设置x轴的显示范围，从0到256

plt.subplot(224)
hist = cv2.calcHist([equilibrated_image], [0], None, [256], [0, 256])
plt.plot(hist, color='gray', linewidth=2, label='gray')  # hist是直方图数据，color='blue'设置线条颜色为蓝色，linewidth=2设置线条宽度为2个像素。
plt.ylabel('Number of Pixels')
plt.xlim([0, 256])  # 设置x轴的显示范围，从0到256

plt.show()
