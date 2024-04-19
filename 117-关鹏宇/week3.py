import cv2
import numpy as np
import time

from matplotlib import pyplot as plt


def bilinearInterpolation(ori_image, resize_height, resize_width):
    ori_h,ori_w,ori_channel = ori_image.shape
    empty_image = np.zeros((resize_height,resize_width,ori_channel),dtype=np.uint8)
    # 计算放大倍率
    ratio_h = ori_h / resize_height
    ratio_w = ori_w / resize_width
    for i in range(ori_channel):
        for resize_y in range(resize_height):
            for resize_x in range(resize_width):
                # 计算原图坐标  图像偏移使几何中心重合  (ori_X + 0.5) = (dst_X + 0.5) * ratio
                ori_img_x = (resize_x + 0.5) * ratio_w - 0.5
                ori_img_y = (resize_y + 0.5) * ratio_h - 0.5
                # 原图坐标大概率不为整数，转换为两侧点
                ori_img_x0 = int(np.floor(ori_img_x))   #np.floor 向下取整
                ori_img_x1 = min(ori_img_x0 + 1, ori_w - 1) #使用min防止边缘越界
                ori_img_y0 = int(np.floor(ori_img_y))
                ori_img_y1 = min(ori_img_y0 + 1, ori_h - 1)
                # 计算插值 (灰度值)
                f_r1 = (ori_img_x1-ori_img_x) * ori_image[ori_img_y0,ori_img_x0,i] + (ori_img_x-ori_img_x0) * ori_image[ori_img_y0,ori_img_x1,i]
                f_r2 = (ori_img_x1-ori_img_x) * ori_image[ori_img_y1,ori_img_x0,i] + (ori_img_x-ori_img_x0) * ori_image[ori_img_y1,ori_img_x1,i]
                empty_image[resize_y,resize_x,i] = (ori_img_y1-ori_img_y) * f_r1 + (ori_img_y-ori_img_y0) * f_r2
    return empty_image

def sobelEdge(image):
    '''
    Sobel函数求完导数后会有负值，还有会大于255的值。
    而原图像是uint8，即8位无符号数(范围在[0,255])，所以Sobel建立的图像位数不够，会有截断。
    因此要使用16位有符号的数据类型，即cv2.CV_16S。
    '''
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    '''
    在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。
    否则将无法显示图像，而只是一副灰色的窗口。
    dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])  
    其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
    '''
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    '''
    由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
    。其函数原型为：
    dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])  
    其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
    gamma是加到最后结果上的一个值。
    '''
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst







################## 双线性插值 ########################
# img = cv2.imread("../../lenna.png")
# start_time = time.time()
# resize_img = bilinearInterpolation(img, 700, 700)
# end1_time = time.time()
# resize_img2 = cv2.resize(img, (700, 700), interpolation=cv2.INTER_LINEAR)
# end2_time = time.time()
# print("bilinearInterpolation time:", end1_time - start_time)
# print("resize time:", end2_time - end1_time)
# cv2.imshow("resize_img2", resize_img2)
# cv2.imshow("resize_img", resize_img)
# cv2.waitKey(0)


################## 直方图均衡化 ########################
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # 灰度图像直方图均衡化
# dst_img = cv2.equalizeHist(gray_img)
# cv2.imshow("Histogram Equalization", np.hstack([gray_img, dst_img]))
# cv2.waitKey(0)
# # 直方图
# hist = cv2.calcHist([dst_img],[0],None,[256],[0,256])
# plt.figure()
# plt.hist(dst_img.ravel(), 256)
# plt.show()

################## sobel边缘提取 ########################
img = cv2.imread("../../lenna.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = sobelEdge(img)

cv2.imshow("Result", dst)
cv2.waitKey(0)
