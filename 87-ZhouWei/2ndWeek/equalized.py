import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('lenna.png')
row, col = image.shape[:2]


# -----------equalizeHist ----------
(b, g, r) = cv2.split(image)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayH = cv2.equalizeHist(gray)



fig = plt.figure(figsize=(20, 14))
chans = cv2.split(image)
colors = ("b", "g", "r")

fig.add_subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")

ax2 = fig.add_subplot(2, 4, 2)
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    ax2.plot(hist, color=color)
    ax2.set_xlim([0, 256])
    ax2.set_title('RGB Histogram')
    ax2.set_xlabel("Bins")
    ax2.set_ylabel("Pixels Num")

fig.add_subplot(2, 4, 3)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("equalizeHist")

hchans = cv2.split(result)
ax4 = fig.add_subplot(2, 4, 4)
for (chan, color) in zip(hchans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    ax4.plot(hist, color=color)
    ax4.set_xlim([0, 256])
    ax4.set_title('RGB equalize Histogram')
    ax4.set_xlabel("Bins")
    ax4.set_ylabel("Pixels Num")

fig.add_subplot(2, 4, 5)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.title("gray")

ax6 = fig.add_subplot(2, 4, 6)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
ax6.plot(hist)
ax6.set_xlim([0, 256])
ax6.set_title('Gray  Histogram')
ax6.set_xlabel("Bins")
ax6.set_ylabel("Pixels Num")

fig.add_subplot(2, 4, 7)
plt.imshow(cv2.cvtColor(grayH, cv2.COLOR_BGR2RGB))
plt.title("gray equalize")

ax8 = fig.add_subplot(2, 4, 8)
hist = cv2.calcHist([grayH], [0], None, [256], [0, 256])
ax8.plot(hist)
ax8.set_xlim([0, 256])
ax8.set_title('Gray equalize Histogram')
ax8.set_xlabel("Bins")
ax8.set_ylabel("Pixels Num")

plt.show()
