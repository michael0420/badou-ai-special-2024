#主要用做图像相似度比较
#二进制数据如何比较相似度？
#汉明距离:指的是这两个数字的二进制位不同的位置的数目
import cv2

#均值哈希
#图片缩放（8*8,保留结构除去细节），灰度化，求平均值，比较（像素值大于均值记为1，否则记为0），生成hash（将上一步的0和1组合起来就是这个图片的指纹）
#对比hash，计算汉明距离，距离越小，图片越相似(按照经验来说小于5-8就算两个图片一样了)
img  = cv2.imread("lenna.png")
img_resize = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
gray_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

sum = 0
hashstr = ''
for i in range(8):
    for j in range(8):
        sum = sum + gray_img[i][j]

avg = sum/64

for i in range(8):
    for j in range(8):
        if gray_img[i][j] > avg:
            hashstr = hashstr +'1'
        else:
            hashstr = hashstr +'0'

print(hashstr)

#差值hash
#缩放为8*9，转化为灰度图，比较：像素值大于后一个像素值就为1，否则记为0，本行不与下一行对比，每行9个像素，八个差值，共64位
#生成hash，对比hash（汉明距离）

img_resize_chazhi = cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
gray_img_chazhi = cv2.cvtColor(img_resize_chazhi, cv2.COLOR_BGR2GRAY)

hash = ''

for i in range(8):
    for j in range(8):
        if gray_img_chazhi[i][j] > gray_img_chazhi[i][j+1]:
            hash = hash +'1'
        else:
            hash = hash +'0'

print(hash)









