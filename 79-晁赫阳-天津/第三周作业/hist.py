import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist(img):
    channels = img.shape[-1] if len(img.shape) == 3 else 1
    if channels == 3 :
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        result = cv2.merge((bH, gH, rH))
        # Plot histogram
        plt.figure(figsize=(10, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
        plt.title('Equalized Image')
        plt.axis('off')  # Turn off axis

        plt.subplot(2, 2, 2)
        plt.hist(rH.ravel(), 256, [0, 256], color='r')
        plt.title('Red Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.subplot(2, 2, 3)
        plt.hist(bH.ravel(), 256, [0, 256], color='b')
        plt.title('Blue Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

        plt.subplot(2, 2, 4)
        plt.hist(gH.ravel(), 256, [0, 256], color='g')
        plt.title('Green Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()
    else:
        dst = cv2.equalizeHist(img)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.figure()
        plt.hist(dst.ravel(), 256)
        plt.show()

if __name__ == '__main__':
    img = cv2.imread('0216.jpg')
    hist(img)
