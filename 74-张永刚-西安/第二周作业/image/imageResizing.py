"""
@author: 74+张永刚
实现临近插值法
"""
import cv2
import numpy as np

# 创建一个处理图像放大缩小的方法
def imgResizing(img,h,w):
    # 获取到图像
    hide,wide,channels = img.shape
    # 输出看一下获取到的是什么内容
    # print(hide)
    # print(wide)
    # print(channels)
    #创建一个空白图像
    emptyImage = np.zeros([h,w,channels],np.uint8)
    sh = h/hide
    sw = w/wide
    for i in range(h):
        for j in range(w):
            # int 转换为整数，想下取整 +0.5 手动四舍五入 ，，，，，，，，rounding mode，IEEE754
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            # 四舍五入
            x = round(i / sh)
            y = round(j / sw)

            emptyImage[i, j] = img[x,y]
    return emptyImage

# 获取图像
img = cv2.imread("img/lenna.png")
# 调用缩放图片方法并传入需要生成图片的大小
emptyImage = imgResizing(img,800,800)
# 输出放大后的图片
cv2.imshow("nearest interp",emptyImage)
# 输出原始图片
cv2.imshow("image",img)
print(emptyImage.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()

