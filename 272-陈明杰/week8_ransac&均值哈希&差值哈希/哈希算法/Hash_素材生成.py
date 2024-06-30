import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageEnhance


# # 旋转图像，默认是逆时针旋转的
# def Rotate(img_path, angle):
#     img = Image.open(img_path)
#     rotated = img.rotate(angle)
#     return rotated

# 旋转图像
def Rotate(img_path, angle):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 定义旋转中心和缩放因子
    (h, w) = img.shape[:2]
    center = (h // 2, w // 2)
    scale = 1.0
    # 计算旋转后的图像边界
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # M的内容如下：
    # [scale * cos(angle), -scale * sin(angle), tx]
    # [scale * sin(angle), scale * cos(angle), ty]
    scale_cos = np.abs(M[0, 0])
    scale_sin = np.abs(M[0, 1])
    nW = int(h * scale_sin + w * scale_cos)
    nH = int(h * scale_cos + w * scale_sin)
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    rotated = cv2.warpAffine(img, M, (nW, nH))
    return rotated



# 色度增强/减弱
def enhance_color(img):
    enh_col = ImageEnhance.Color(img)
    color = 0.5
    return enh_col.enhance(color)


# 模糊操作
def GaussianBlur(img):
    # 高斯模糊，也叫高斯滤波
    res = cv2.GaussianBlur(img, (15, 15), 0)
    return res


# 锐化操作
def sharp(img):
    kernel = np.array([[0, -1, 0], [-1, 5.1, -1], [0, -1, 0]])
    # 利用特殊的卷积核进行卷积，达到锐化的效果
    return cv2.filter2D(img, -1, kernel=kernel)


# 调节对比度和亮度
def adjust_contrast_brightness(src1):
    h, w, channel = src1.shape
    src2 = np.zeros([h, w, channel], src1.dtype)
    # 参数1：src1 - 第一个输入数组或图像。
    # 参数2：a - 第一个数组对应项的权重。
    # 参数3：src2 - 第二个输入数组或图像，它和src1应具有相同的尺寸和类型。
    # 参数4：1 - a - 第二个数组对应项的权重。通常，权重之和应为1（即 a + (1 - a) = 1），
    # 以确保结果图像在强度范围内。
    # 参数5：g - 可选参数，表示加到加权和上的标量值。
    return cv2.addWeighted(src1, 1.2, src2, 0.2, 1)


# 缩放图片
def resize(img):
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)


# 修改图片的亮度
def light(img):
    # np.clip() 函数是 NumPy 库中的一个函数，用于将数组中的值限制在指定的最小值和最大值之间
    # 最大最小值分别对应第二第三个参数
    return np.uint8(np.clip(1.5 * img + 10, 0, 255))


# 旋转图像
img_path = 'lenna.png'
rotated = Rotate(img_path, 45)
plt.figure()
# 设置matplotlib正常显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.subplot(241)
plt.title('旋转图像')
plt.imshow(rotated)

# 色度增强
enh_col = enhance_color(Image.open('lenna.png'))
plt.subplot(242)
plt.title('色度增强')
plt.imshow(enh_col)

# 高斯模糊
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
blur = GaussianBlur(img)
plt.subplot(243)
plt.title('高斯模糊')
plt.imshow(blur)

# 图像锐化
# img_sharp = sharp(img)
# plt.subplot(244)
# plt.title('图像锐化')
# plt.imshow(img_sharp)
img1 = cv2.imread('lenna.png')
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
cv2.imshow('jt', img1)
cv2.waitKey(0)
img1 = cv2.resize(img1, (int(img1.shape[1]*1.5),int(img1.shape[0]*1.5)), interpolation=cv2.INTER_CUBIC)
img_sharp = sharp(img1)
cv2.imwrite('1.jpg',img_sharp)
cv2.imshow('jt', img_sharp)
cv2.waitKey(0)
plt.subplot(244)
plt.title('图像锐化')
plt.imshow(img_sharp)

# 调节对比度和亮度
adjust_contrast_brightness = adjust_contrast_brightness(img)
plt.subplot(245)
plt.title('调节对比度和亮度')
plt.imshow(adjust_contrast_brightness)

# 缩放图片
img_resize = resize(img)
plt.subplot(246)
plt.title('缩放图片')
plt.imshow(img_resize)

# 修改图片的亮度
light_img = light(img)
plt.subplot(247)
plt.title('修改图片的亮度')
plt.imshow(light_img)

plt.show()
