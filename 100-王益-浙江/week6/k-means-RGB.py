import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread('img/lenna.png')
data =np.float32(img.reshape(-1,3))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags =cv2.KMEANS_RANDOM_CENTERS

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(2,3,1)
plt.imshow(img)
plt.xticks([]), plt.yticks([])
K = 2
for i in range(5):
    compactness,labels,centers = cv2.kmeans(data,K,None,criteria,10,flags)
    centers = np.uint8(centers)
    result_img = centers[labels.flatten()]
    result_img = result_img.reshape(img.shape)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    plt.subplot(2,3,i+2), plt.imshow(result_img, 'gray')
    plt.xticks([]), plt.yticks([])
    K = K*2
plt.show()

