import cv2
import numpy as np
def img_tsf(img):
    heigt,width,channels=img.shape  #获取图片的高度、宽度、通道数
    emptyImage=np.zeros((400,400,channels),np.uint8)  #通过zeros函数创建一张数据类型为uint8，800*800的空白图片
    a=400/heigt
    b=400/width   #转换比例，原图与空白图片的高、宽的比例
    for i in  range(400):
        for j in range(400):
            x=int(i/a+0.5)
            y=int(j/b+0.5)#+0.5，达到四舍五入
            emptyImage[i,j]=img[x,y]    #通过比例转换，得到空白图片的新坐标值
    return emptyImage

img=cv2.imread('F:/PNG/lenna.png')    #通过imread函数读取本地图片
tsfImg=img_tsf(img)   #通过上面函数转换图片
print(tsfImg)
print(tsfImg.shape)  #输出转换后的图片的高度、宽度、通道数
cv2.imshow("Nearest neighbor interpolation method",tsfImg)   #显示转换后的图片
cv2.imshow("Original image",img)
cv2.waitKey(0) #函数waitKey无限期地等待一个键事件或等待延迟毫秒,当x>0，waitkey返回在x时间内按下的按键的ASCII值;当x=0，waitkey表示永久等待，直到有键按下





