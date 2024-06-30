import cv2
import numpy as np
import matplotlib.pyplot as plt
img  = cv2.imread("../../lenna.png")

data = img.reshape((-1,3))
data = np.float32(data)
print(data)

#停止条件
cer = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10 ,1)

#
flags = cv2.KMEANS_RANDOM_CENTERS

compactness , labels , centers = cv2.kmeans(data,2,None,cer,10,flags)


centers = np.uint8(centers)
res = centers[labels.flatten()]

dst2 = res.reshape((img.shape))

dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)

plt.subplot(1,1,1)
plt.imshow(dst2, 'gray')
plt.show()





