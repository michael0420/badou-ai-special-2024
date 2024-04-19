# 1实现双线性插值、2实现直方图均衡化、3实现sobel边缘检测
import cv2
import numpy as np
from matplotlib import pyplot as plt
# 1实现双线性插值
def function(img,dst_h,dst_w):
    src_h,src_w,channel = img.shape
    if src_w == dst_w and src_h == dst_h:
        return img
    # 缩放比例
    scale_h = float(src_h)/dst_h
    scale_w = float(src_w)/dst_w
    # 目标图初始化
    dst_img = np.zeros((dst_h,dst_w,channel),np.uint8)
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 几何中心对称
                src_x = (dst_w + 0.5) * scale_w - 0.5
                src_y = (dst_h + 0.5) * scale_h - 0.5
                # 防止越界
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0+1,src_w-1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0+1,src_h-1)
                # 求点(src_x,srx_y)的像素值、需要根据它的临近四点(x0,y0)、(x0,y1)、(x1,y0)、(x1,y1)计算出
                # (x1-x/x1-x0)*(y0,x1) + (x-x0)/(x1-x0)* (y0,x1),因为x1-x0=1分母为1
                # (x1-x)*(y0,x0) + (x-x0)* (y0,x1)
                temp0 = (src_x1-src_x) * img[src_y0, src_x0, i] + (src_x-src_x0) * img[src_y0, src_x1, i]
                # (x1-x)*(y1,x0) + (x-x0)* (y1,x1)
                temp1 = (src_x1-src_x) * img[src_y1, src_x0, i] + (src_x-src_x0) * img[src_y1, src_x1, i]
                # 点(src_x,srx_y)的像素值 = (y1-y)*temp0 + (y-y0)* temp1
                dst_img[dst_y,dst_x,i] = int((src_y1-src_y)*temp0 + (src_y-src_y0) * temp1)

    cv2.imshow("dst_img", dst_img)

# 2实现直方图均衡化
def img_hist(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 直方图均衡化
    j_hist = cv2.equalizeHist(gray)
    # 得到直方图
    gary_hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    # rgb三通道直方图
    (r,g,b) = cv2.split(img)
    r = cv2.equalizeHist(r)
    g = cv2.equalizeHist(g)
    b = cv2.equalizeHist(b)
    rgb_hist = cv2.merge((r,g,b))
    # 展示图像
    plt.figure()
    plt.subplot(231)
    plt.hist(gary_hist.ravel(), 256)
    plt.subplot(232)
    plt.hist(j_hist.ravel(), 256)
    plt.subplot(233)
    plt.hist(rgb_hist.ravel(), 256)
    plt.subplot(234)
    plt.imshow(gray, cmap="gray")
    plt.subplot(235)
    plt.imshow(j_hist, cmap="gray")
    plt.subplot(236)
    l = plt.imread(path)
    plt.imshow(l)
    plt.show()

# 3实现sobel边缘检测
def sobel(img):
    # 转16位数据
    x = cv2.Sobel(img,cv2.CV_16S,1,0)
    y = cv2.Sobel(img,cv2.CV_16S,0,1)
    # 转回8位数据
    absx = cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    # 组合x方向和y方向
    result = cv2.addWeighted(absx,0.5,absy,0.5,0)
    cv2.imshow("absx",absx)
    cv2.imshow("absy",absy)
    cv2.imshow("result",result)


# -调用自定义函数-------------------------------------------------------------
# 双线插值
img = cv2.imread("lenna.png")
function(img, 300, 300)
# 直方图
img_hist("lenna.png")
# 边缘检测
sobel(img)

cv2.waitKey(0)
cv2.destroyAllWindows()
