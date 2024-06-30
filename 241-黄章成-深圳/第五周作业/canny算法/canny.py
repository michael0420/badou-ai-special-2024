import cv2

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是阈值1；
第三个参数是阈值2。
'''

img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 100: 这是 Canny 边缘检测的低阈值参数。低阈值用于定义边缘像素的梯度值，低于这个阈值的边缘会被丢弃。
# 200: 这是 Canny 边缘检测的高阈值参数。高阈值用于定义强边缘像素的梯度值，高于这个阈值的像素会被认为是强边缘。
cv2.imshow("canny", cv2.Canny(gray, 100, 200))
cv2.waitKey()
cv2.destroyAllWindows()