import cv2
import numpy as np
from matplotlib import pyplot as plt
#
def equalizeHistogram(img_path):
    img=cv2.imread(img_path)
    gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # result
    res=cv2.equalizeHist(gray)

    hist=cv2.calcHist([res],[0],None,[256],[0,256])
    plt.figure()
    plt.hist(gray.ravel(), 256)
    plt.hist(res.ravel(), 256)

    plt.show()
    # 均衡化前后对比
    cv2.imshow('equalized',np.hstack((gray,res)))
    cv2.waitKey()

def equalizeHistogram2(img_path):
    img = cv2.imread(img_path)
    b,g,r = cv2.split(img)
    dst_r=cv2.equalizeHist(r)
    dst_g=cv2.equalizeHist(g)
    dst_b=cv2.equalizeHist(b)
    res=cv2.merge([dst_b,dst_g,dst_r])
    cv2.imshow('res',np.hstack((img,res)))
    cv2.waitKey()

    plt.figure()
    plt.hist(img.ravel(),256)
    plt.hist(res.ravel(),256)
    plt.show()


equalizeHistogram('../Lenna.jpg')
# equalizeHistogram2('../Lenna.jpg')

