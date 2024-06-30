import cv2
import numpy as np
from matplotlib import pyplot as plt


# 自己写drawMatchsKnn
def cv2_drawMatchsKnn(img1, kp1, img2, kp2, good):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 将两张输入图像分别放置在emp图像的左右两侧，以便后续在上面绘制匹配线条。
    emp = np.zeros((max(h1, h2), w1 + w2, 3),np.uint8)
    emp[:h1, :w1] = img1
    emp[:h2, w1:w1 + w2] = img2

    # 从 good中获取匹配点对的索引，queryIdx 表示在第一张图像中的索引，trainIdx 表示在第二张图像中的索引。
    p1 = [kpp.queryIdx for kpp in good]
    p2 = [kpp.trainIdx for kpp in good]



    # 从特征点列表得到匹配点的位置。 .pt:代表位置。  第二章图要加上第一张的宽
    post1 = np.int32([kp1[pp].pt for pp in p1])     #np.int32是为了变成整型，用于cv2.line.cv2.line需要是整型
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)   # 且元组和列表不能相连

    # 使用 cv2.line 函数在emp图像上绘制了匹配点对之间的连线
    '''
    zip 将对象中对应的元素霸道成一个个元组。 
    print(zip)：显示的是包所在的地址
    print(list(zip)): 显示的是包里的内容

    zip将post1 和post2 打包成一个元组。
    '''
    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(emp, (x1, y1), (x2, y2), color=(0,255,0))

    '''
    flag: 表示窗口大小是自动设置还是可调整
         cv2.WINDOW_NORMAL ：允许手动更改
         cv2.WINDOW_AUTOSIZE(default)  ： 自动设置
         cv2.WINDOW_FULLSCREEN ：全屏
    '''
    cv2.namedWindow('match', cv2.WINDOW_NORMAL)
    cv2.imshow('match', emp)  # cv2直接imshow，大小无法调整


img1 = cv2.imread('iphone1.png')
img2 = cv2.imread('iphone2.png')


sift = cv2.SIFT_create()

# 分别对两张图像进行特征检测和特征描述
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

'''
BFmatcher:暴力匹配。 Brute-Force
cv2.BFMatcher() 
（）中参数:  
           cv2.NORM.L2: 默认  SIFT,SURF
           cv2.NORM_HAMMING 对于基于二进制字符串的描述符，用汉明距离


matches 返回的是DMatch数据结构的列表。包含：
    distance：描述符之间的距离
    queryIdx:主动匹配的描述符组中描述符的索引
    trainIdx:被匹配的描述符族中描述符的索引
    imgIdx：目标图像的索引

BFMatcher.match(): 返回最佳匹配
BFMatcher.knnMatch():返回k个最佳匹配，k要自己定


'''

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)  #返回值为m,n

#设定一个距离比例
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
'''
运用 cv.drawMatchesKnn()：绘制出匹配项，水平堆叠

good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2= cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
plt.axis('off')
plt.show()
'''

cv2_drawMatchsKnn(img1, kp1, img2, kp2, good[:30])

cv2.waitKey()
cv2.destroyAllWindows()
