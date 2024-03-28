
import matplotlib.pyplot as plt
import cv2
import numpy as np

print("image process")
def bgrToGray(ori):
    row, col = ori.shape[:2]
    img_gray = np.zeros([row, col], ori.dtype)
    for i in range(row):
        for j in range(col):
            temp = ori[i, j]
            img_gray[i, j] = int(temp[0]*0.11 + temp[1] * 0.59 + temp[2] * 0.3)
    return img_gray

def grayToBinary(gray):
    row, col = gray.shape[:2]
    img_binary = np.zeros([row, col], gray.dtype)
    for i in range(row):
        for j in range(col):
            if (gray[i, j] < 128):
                img_binary[i, j] = 0
            else:
                img_binary[i, j] = 255
    return img_binary

def nearestInsertPixels(ori):
    row, col, channels = ori.shape
    img_binary = np.zeros([2*row, 2*col, channels], ori.dtype)
    for i in range(2*row):
        for j in range(2*col):
            x = int(i/2-1+0.5)
            y = int(j/2-1+0.5)
            img_binary[i, j] = ori[x, y]

    return img_binary

imgOriginal = cv2.imread("img.png")

imgGray = bgrToGray(imgOriginal)
imgGrayCV2 = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

imgBinary = grayToBinary(imgGrayCV2)
#imgBinary = np.where(imgGrayCV2 > 125, 255, 0).astype(np.uint8)
ret, imgBinaryCV2 = cv2.threshold(imgGrayCV2, 127, 255, cv2.THRESH_BINARY)

incrasedPixels = nearestInsertPixels(imgOriginal)

fig = plt.figure(figsize=(10, 7))

fig.add_subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2RGB))
plt.title("Original")

fig.add_subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(imgGray, cv2.COLOR_BGR2RGB))
plt.title("Gray by self define function")

fig.add_subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(imgGrayCV2, cv2.COLOR_BGR2RGB))
plt.title("Gray by cv2")

fig.add_subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(incrasedPixels, cv2.COLOR_BGR2RGB))
plt.title("Nearest insert pixels")

fig.add_subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(imgBinary, cv2.COLOR_BGR2RGB))
plt.title("Binary by self define function")

fig.add_subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(imgBinaryCV2, cv2.COLOR_BGR2RGB))
plt.title("Binary by CV2")

plt.show()
