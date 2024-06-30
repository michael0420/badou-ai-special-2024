'''
sift方法图像的关键点检测，对于尺度、旋转和亮度都具有不变性
一般建立sift生成器是用函数cv2.xfeatures2d.SIFT_create()。
opencv版本是4.5.5，建议4.0以上时使用cv2.SIFT_create()
'''
import cv2

img = cv2.imread('lenna.png', 0)
sift = cv2.SIFT_create()  # 实例化,创建了一个对象，后面是对象具有的方法
keypoints, descriptor = sift.detectAndCompute(img, None)  # 直接找到特征点并计算描述符
'''
kp:关键点信息，包括位置，尺度，方向信息
des:关键点描述符，每个关键点对应128个梯度信息的特征向量
'''
# keypoints = sift.detect(img, None)  # 找到特征点即可，不需描述子即可绘图
# print(keypoints, descriptor, type(keypoints), type(descriptor))

# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS()  # 对图像的每个关键点都绘制了圆圈和方向。
img = cv2.drawKeypoints(image=img, keypoints=keypoints, outImage=img,
                        color=(255, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
'''
flags:绘图功能的标识设置
cv2.DRAW_MATCHES_FLAGS_DEFAULT:创建输出图像矩阵，使用现存的输出图像匹配对和特征点，对每一个关键点，对每一个关键点只绘制中间点.
cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG:不创建输出图像矩阵，而是在输出图像上绘制匹配对。
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS:对每一个特征点绘制带大小和方向的关键点图形。或者flags=4
cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS:单点的特征点不被绘制。
'''
# img = cv2.drawKeypoints(img, keypoints, img, color=(5, 122, 25))  # 只带标准圆绘制关键点位置，但没有大小。color='red'会报错
cv2.imshow('sift keypoints', img)
cv2.waitKey()
cv2.destroyAllWindows()
