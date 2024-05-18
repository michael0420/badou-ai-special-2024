"""
Canny边缘检测：优化的程序(使用调节杠观察不同阈值的结果)
这个实验为了证明阈值大小对边缘检测结果影响很大，随着阈值增大保留的边缘越来越少
"""
import cv2


def canny_threshold(low_threshold):
    detected_edges = cv2.Canny(img_gray,
                               low_threshold,
                               low_threshold*ratio,
                               apertureSize=kernel_size)

    # 用图片原始颜色添加到检测的边缘上。
    # 按位“与”操作。对于每个像素,将两幅输入图像相应位置的像素值分别进行按位“与”运算,输出的结果图像的对应像素值即为这两幅输入图像对应像素值的按位与结果。
    # 按位“与”：只有两个值一样才保留，所以两张一样的图片加上掩模按位与就是只保留掩模的部分
    # mask 是可选参数，如果指定了掩膜，则只对掩膜对应位置的像素进行按位“与”操作。函数的返回值表示按位“与”运算的结果。
    dst = cv2.bitwise_and(img, img, mask=detected_edges)  # 如果不加mask输出就是原图
    cv2.imshow('canny result', dst)


low_threshold = 0
max_threshold = 100  # 不是高阈值，而是低阈值最大能拉到的值
ratio = 3  # 高阈值是低阈值的ratio倍
kernel_size = 3

img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 第二个参数flag默认值是cv2.WINDOW_AUTOSIZE（窗口大小根据显示图像的大小自动调整，用户不能手动改变窗口大小。）
cv2.namedWindow("canny result", cv2.WINDOW_AUTOSIZE)  # cv2.WINDOW_NORMAL可以手动改变窗口大小

'''
下面是第二个函数，cv2.createTrackbar()
共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名
'''
cv2.createTrackbar('min_threshold', 'canny result', low_threshold, max_threshold, canny_threshold)
canny_threshold(0)
if cv2.waitKey(0) == 27:  # 在ASCII表中，ESC键的编码是27
    cv2.destroyAllWindows()
