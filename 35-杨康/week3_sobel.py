import cv2
from matplotlib import pyplot as plt

img_gray = cv2.imread('lenna.png',0)

# Sobel函数求完导数后会有负值，还有会大于255的值,要使用16位有符号的数据类型，即cv2.CV_16S
x = cv2.Sobel(img_gray,cv2.CV_16S,1,0)  # 对x求一阶导
y = cv2.Sobel(img_gray,cv2.CV_16S,0,1)  # 对y求一阶导

#转换回原数据类型
absx = cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)
#加权重
dst = cv2.addWeighted(absx,0.5,absy,0.5,0)  # 0  gamma是加到最后结果上的一个值

#显示图形
titles = ['lenna','Sobel_x','Sobel_y','Sobel']
images = [img_gray,absx,absy, dst]
for i in range(4):
    plt.subplot(1,4,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

