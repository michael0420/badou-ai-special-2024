# 图像由彩色图转化为灰度图常用的计算方式有以下三种：
# ① 权重方法：Gray = R0.3+G0.59+B0.11
# ② 平均值法：Gray = (R+G+B) / 3
# ③ 仅取某一通道：Gray  = R
import cv2
import numpy as np
#当数组元素总数过大时，设置显示的数字位数，其余用省略号代替(当数组元素总数大于设置值，控制输出值得个数为6个，当数组元素小于或者等于设置值得时候，全部显示)，当设置值为sys.maxsize(需要导入sys库)，则会输出所有元素
#np.set_printoptions(threshold=100000000)   #用于控制Python中小数的显示精度。
image = cv2.imread(r'C:/Users/Administrator/Pictures/hongkong.png', 1)
def rgb_to_rgba(rgb_color, alpha=255):
    return rgb_color + (alpha,)

#4通道加了透明度
rgba_color = rgb_to_rgba( image,64)
cv2.imwrite('C:/Users/Administrator/Pictures/hongkong_rgba_color.png', rgba_color)
print(rgba_color)  # 输出 (255, 0, 0, 128)
print(image)
print(type(image))
higth, weigth,channel = image.shape[:3]
print(higth,weigth,channel)
#初始化一个空的矩阵
image_gray = np.zeros([higth, weigth], image.dtype)
print(image_gray)
for i in range(higth):
    for j in range(weigth):
        m = image[i,j]
        image_gray[i,j] = int(m[0] *.11+ m[1]*0.59+m[2]*0.30)
#变灰系数生成灰度图片
cv2.imwrite('C:/Users/Administrator/Pictures/hongkong_gray2.png', image_gray)
#通过方法直接转换成灰度
img_gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image show gray",img_gray2)
cv2.imwrite('C:/Users/Administrator/Pictures/hongkong_gray.png', img_gray2)
# def matrix_equal(matrix1, matrix2):
#     # 判断矩阵的行数和列数是否相等
#     if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
#         return False
#
#     #遍历矩阵的每一个元素，比较对应位置的元素是否相等
#     for i in range(len(matrix1)):
#         for j in range(len(matrix2)):
#             print("----------",i,j)
#             if matrix1[i][j] != matrix2[i][j]:
#                 return False
#
#     return True
import pandas as pd
print(len(image_gray),len(img_gray2))
# equal = matrix_equal(image_gray,img_gray2)
# print(equal)
from matplotlib import pyplot as plt
#转为二值
import cv2
def convert_image_to_binary(image_path, threshold=100):
    # 读取图像
    image_11 = cv2.imread(image_path, 0)
    # 二值化图像
    _, binary_image = cv2.threshold(image_11, threshold, 255, cv2.THRESH_BINARY)
    # 返回二值图像
    return binary_image
#image_111 = cv2.imread(r'C:/Users/Administrator/Pictures/dogcat.jpg', 0)
binary_image = convert_image_to_binary(r'C:/Users/Administrator/Pictures/hongkong.png')
cv2.imwrite(r'C:/Users/Administrator/Pictures/hongkong_binary.png', binary_image)



#上下采样
height1 =1000
width1 =800
import cv2
import numpy as np
def function(img):
    height,width,channels =img.shape
    emptyImage=np.zeros((height1,width1,channels),np.uint8)    #建立一个空图像
    sh=height1/height
    sw=width1/width
    for i in range(height1):
        for j in range(width1):
            x=int(i/sh)
            y=int(j/sw)
            emptyImage[i,j]=img[x,y]
    return emptyImage

img=cv2.imread(r"C:/Users/Administrator/Pictures/hongkong.png")
print(img.shape)
zoom=function(img)
cv2.imwrite(r'C:/Users/Administrator/Pictures/hongkong8000.png', zoom)
print(zoom.shape)
#cv2.imshow("nearest interp",zoom)
#cv2.imshow("image",img)
cv2.waitKey(0)
with open("filename.txt", "w") as file:
    # file.write("_________________________________________\n")
    # file.write(str(image))
    file.write(str(image[0]))
    file.write("_________________________________________\n")
    file.write(str(image[1]))
    file.write("_________________________________________\n")
    file.write(str(len(image)))
    file.write("_________________________________________\n")
    file.write(str(len(image[0][0])))
    file.write("_________________________________________\n")
    # file.write("\n_________________________________________\n")
    #
    # file.write("\n_________________________________________\n")
    # file.write(str(image_gray))
    # file.write("\n_________________________________________\n")
    # file.write(str(image.shape[0]))
    # file.write("\n")
    # file.write(str(image.shape[1]))
    # file.write("\n")
    # file.write(str(image.shape[2]))
    # file.write(str(image_gray.shape[0]))
    # file.write("\n")
    # file.write(str(image_gray.shape[1]))
    # file.write("\n")
    # file.write(str(image_gray.shape[2]))
    # file.write("\n_________________________________________\n")
#cv2.imshow("image show gray",image_gray)
cv2.imwrite('C:/Users/Administrator/Pictures/hongkong13.png', image_gray)
# 使用plt.subplot来创建小图.
fig = plt.figure(figsize=(100,100),dpi=600)  # 初始化一张画布

plt.subplot(131), plt.imshow(image.astype('uint8')), plt.title('Original Image')
plt.subplot(132), plt.imshow(img_gray2.astype('uint8'), cmap='gray'), plt.title('Global Thresholding')
plt.subplot(133), plt.imshow(rgba_color.astype('uint8')), plt.title('Global Thresholding')
plt.show()
plt.axis('off')
cv2.waitKey(0)
cv2.destroyAllWindows()