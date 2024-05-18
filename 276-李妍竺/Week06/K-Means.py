import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png',0)
print(img.shape)

h,w = img.shape
data = img.reshape(h*w,1)
data = np.float32(data)
print(data)

#停止条件
criteria = cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,12,1.0

#初始中心点：三种方法。
#flags = cv2.KMEANS_RANDOM_CENTERS
#flags = cv2.KMEANS_PP_CENTERS
flags = cv2.KMEANS_USE_INITIAL_LABELS

# K_means
compactness,Lables,centers = cv2.kmeans(data,4,None,criteria,8,flags)

result = Lables.reshape(img.shape[0],img.shape[1])

print('result',result)

#plt.rcParams:用来修改默认属性，包括船体大小，每英寸点数，线条宽度、颜色、样式、字体等
plt.rcParams['font.sans-serif']=['Kaiti']  # 防止中文报错  SimHei:黑体

plt.subplot(121)
plt.imshow(img,cmap='gray')
plt.title('原始图像')
plt.axis('off')

plt.subplot(122)
plt.imshow(result,cmap='gray')
plt.title('K-means图像')
plt.axis('off')

plt.show()
