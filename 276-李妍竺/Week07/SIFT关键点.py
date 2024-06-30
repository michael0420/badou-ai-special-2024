import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#调用sift
sift = cv2.SIFT_create()  # 新版python适配

#寻找关键点与描述子
'''
输入：
image：灰度图或彩色图
mask:可选的淹没图像，可指定区域
kp与des 可选，如果输入了，就不输出了
输出：
kp: 关键点列表
des:特征描述子，形状为(n,128）
'''
keypoints, descriptors = sift.detectAndCompute(gray,None)  #也可用img，为方便用gray

print('kp:',keypoints)
print('des:',descriptors)

#绘制
'''
image:输入图像； keypoint:获取的特征点；outimage:输出图像  color:默认为随机，也可自定义
flags:绘制点的模式
     cv2.DRAW_MATCHES_FLAGS_DEFAULT: 默认值，只绘制特征点的坐标点，是一个个小圆点
     cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS :单点特征点不被绘制
     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: 为每个关键点绘制圆圈与方向
'''
h,w = img.shape[:2]
img_bgr = np.zeros([h,w],img.dtype)
img_gray = np.zeros([h,w],img.dtype)

img_bgr = cv2.drawKeypoints(img,keypoints,img_bgr,color=(),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img_gray = cv2.drawKeypoints(gray,keypoints,img_gray,color=(),flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)



plt.rcParams['font.sans-serif']=['SimHei']
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB)


plt.subplot(221)
plt.imshow(img)
plt.title('原图')
plt.axis('off')

plt.subplot(222)
plt.imshow(img_bgr)
plt.title('彩色特征与方向')
plt.axis('off')


plt.subplot(223)
plt.imshow(img_gray)
plt.title('灰色单点抑制特征')
plt.axis('off')

plt.show()