import cv2

img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(cv2.__version__)

sift = cv2.SIFT_create()
# 获取关键点和描述符
keypoints, descriptor = sift.detectAndCompute(img_gray, None)  # 第二个参数是mask，通常取整张图象

'''
cv2.drawKeypoints 是 OpenCV 中用于在图像上绘制关键点的函数。该函数的参数及其解释如下：
image：
类型：numpy.ndarray
说明：源图像，即要在其上绘制关键点的图像。
keypoints：
类型：list 或 numpy.ndarray
说明：关键点列表或数组，通常是由特征检测器（如 SIFT、SURF、ORB 等）返回的对象。这些对象包含了关键点的位置、大小和方向等信息。
outImage：
类型：numpy.ndarray
说明：输出图像，即绘制了关键点的图像。如果设置为 None，则函数将创建一个新的图像并返回。如果提供了输出图像，则关键点将被绘制在这个图像上。
flags：
类型：int
说明：绘制关键点的选项标志。它可以是以下标志的组合：
cv2.DRAW_MATCHES_FLAGS_DEFAULT：默认值，绘制关键点而不添加任何特殊标记。
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS：对于每个关键点，绘制一个带有大小和方向的圆，并绘制关键点方向上的线段。
cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS：如果设置，则不绘制匹配的关键点。
cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG：如果设置，则匹配和关键点将被绘制在输出图像上。否则，它们将被绘制在一个覆盖在输出图像上的遮罩上。
color：
类型：tuple
说明：用于绘制关键点的颜色，通常是一个BGR格式的元组，例如 (255, 0, 0) 表示蓝色。
params：
类型：dict
说明：包含绘制关键点的其他参数的字典。这个参数通常不被使用，除非你正在使用特定的特征检测器或描述符，并且该检测器或描述符需要额外的参数来绘制关键点。
'''
img = cv2.drawKeypoints(image=img, keypoints=keypoints, outImage=img,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(204, 237, 199))

cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
