import cv2
import numpy as np
'''
img = cv2.imread('lenna.png',0)
cv2.imshow("canny",cv2.Canny(img,200,350))
cv2.waitKey()
cv2.destroyAllWindows()
'''

'''
Canny边缘检测：优化的程序
'''

def CannyThreshold(LowThreshold):
    detected_edges = cv2.GaussianBlur(gray,(5,5),0)
    detected_edges = cv2.Canny(detected_edges,LowThreshold,LowThreshold*ratio,apertureSize=kernel_size)

    # 用原始颜色添加到检测的边缘上
    dst = cv2.bitwise_and(img,img,mask=detected_edges) #mask:掩膜
    cv2.imshow('canny demo',dst)

LowThreshold = 0
Max_LowThreshold = 120
ratio = 3
kernel_size =3

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.namedWindow('canny demo')


#调节杠杆
cv2.createTrackbar('Min Threshold','canny demo',LowThreshold,Max_LowThreshold,CannyThreshold)
CannyThreshold(0)  #初始值
if cv2.waitKey(0) == 27:  #wait for ESC key to exit cv2
    cv2.destroyAllWindows()


#设置调节杠,
'''
第二个函数，cv2.createTrackbar()

第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名
'''