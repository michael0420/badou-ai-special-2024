import cv2
import numpy as np
import matplotlib.pyplot as plt


src_img = cv2.imread('lenna.png', 1)
src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
src_blue, src_green, src_red = cv2.split(src_img)

hist_gray = cv2.equalizeHist(src_gray)
hist_blue = cv2.equalizeHist(src_blue)
hist_green = cv2.equalizeHist(src_green)
hist_red = cv2.equalizeHist(src_red)
hist_img = cv2.merge((src_blue, src_green, src_red))
dst_img = np.hstack((src_gray, hist_gray))
cv2.imshow('Histogram_Equalization', dst_img)
cv2.waitKey(0)


hist_gray_value = cv2.calcHist([src_gray], [0], None, [256], [0, 255])
fig = plt.figure()
ax00 = fig.add_subplot(221)
plt.plot(hist_gray_value, color='k')

ax01 = fig.add_subplot(222)
channel = ('b', 'g', 'r')
for i, col in enumerate(channel):
    hist_channel_value = cv2.calcHist([src_img], [i], None, [256], [0, 255])
    plt.plot(hist_channel_value, color=col)

ax03 = fig.add_subplot(223)
plt.hist(np.ravel(hist_gray), 256, color='k')

ax04 = fig.add_subplot(224)
plt.hist(np.ravel(hist_blue), 256, color='b')
plt.hist(np.ravel(hist_green), 256, color='g')
plt.hist(np.ravel(hist_red), 256, color='r')

plt.show()