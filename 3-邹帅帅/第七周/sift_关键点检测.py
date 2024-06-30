import cv2
import numpy as np

img = cv2.imread(r'E:\image\lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SIFT_create()是创建SIFT对象的函数，位于 cv2.xfeatures2d 模块中
'''
cv2.xfeatures2d 是OpenCV库中的一个模块，专门用于处理特征检测与描述。
SIFT_create() 是该模块中的一个函数，用于实例化一个SIFT类的对象。这个函数不需要传入参数，它会根据SIFT算法的默认设置来配置特征检测器。
'''
sift = cv2.xfeatures2d.SIFT_create()

'''
keypoints, descriptors = sift.detectAndCompute(gray, None) 
这段代码的含义是使用之前创建的SIFT对象（sift）来检测图像中的关键点（特征点）并计算它们的描述符。
gray：这是一个灰度图像，作为detectAndCompute方法的输入。因为SIFT算法通常需要在灰度图像上操作，所以在调用此方法前，原始彩色图像一般会被转换为灰度图。
None：这里是掩码参数，用于指定图像中感兴趣的区域。如果传入None（默认值），则考虑整个图像。
detectAndCompute() 方法执行以下两个操作：
detect：在输入的灰度图像中检测关键点。这些关键点是图像中具有独特属性的位置，比如边缘、角点等，它们对于不同的尺度和旋转相对稳定。
compute：为每个检测到的关键点计算一个描述符。描述符是一个向量，用于表征关键点周围的视觉特征，使得即便在不同图像中，相似的关键点也能通过比较它们的描述符来进行匹配。

keypoints：返回的是一个列表，包含了检测到的所有关键点的信息。每个关键点通常包含位置（x, y坐标）、尺度、方向等属性。
descriptors：这是一个二维数组（通常是numpy数组），每一行对应一个关键点的描述符。这些描述符是用于后续特征匹配的关键数据。
'''
keypoints, descriptors = sift.detectAndCompute(gray, None)

'''
#  cv2.drawKeypoints()是OpenCV库中的一个函数，用于在图像上绘制检测到的关键点（keypoints）
参数
1.image: np.ndarray 类型，这是输入的原始图像，在其上将会绘制关键点。图像应当是8位的灰度图像或彩色图像。
2.keypoints: list 类型，包含了检测到的关键点集合。每个关键点通常是一个具有位置信息（坐标）和其他属性（如大小、方向）的对象。在OpenCV中，这些通常是cv2.KeyPoint对象的列表。
3.outImage (可选): np.ndarray 类型，默认为None。这是输出图像，即绘制了关键点后的图像。如果设为None，则函数会在输入图像上直接绘制，否则会在提供的图像副本上操作。
4.color (可选): 用来绘制关键点的颜色。可以是BGR格式的元组，如(255, 0, 0)表示蓝色。默认值可能会因OpenCV版本不同而变化，一些版本中默认使用随机颜色。
5.flags（可选）: 控制关键点绘制方式的标志。例如，cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS会包含大小和方向信息。
'''

img = cv2.drayKeypoints(img, keypoints, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('keypoints', img)
cv2.waitKer(0)
cv2.destroyAllWindows()