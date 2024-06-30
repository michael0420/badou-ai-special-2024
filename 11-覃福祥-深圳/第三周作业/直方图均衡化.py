#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt
'''
equalizeHist��ֱ��ͼ���⻯
����ԭ�ͣ� equalizeHist(src, dst=None)
src��ͼ�����(��ͨ��ͼ��)
dst��Ĭ�ϼ���
'''

img=cv2.imread('lenna.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray",gray)

# �Ҷ�ͼ��ֱ��ͼ���⻯
dst=cv2.equalizeHist(gray)

#ֱ��ͼ
hist=cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(),256) #plt.hist(src,pixels)����ֱ��ͼ,src:����Դ��ע������ֻ�ܴ���һά���飬ʹ��src.ravel()���Խ���άͼ����ƽΪһά����,pixels:���ؼ���һ������256��
plt.show()
cv2.imshow("Histogram Equalization",np.hstack([gray,dst]))


#��ɫͼ��ֱ��ͼ���⻯
# img=cv2.imread('lenna.png',1)
# cv2.imshow('src',img)
#
# #��ɫͼ����⻯����Ҫ�ֽ�ͨ������ÿһ��ͨ�����⻯
# (b,g,r)=cv2.split(img)
# bH=cv2.equalizeHist(b)
# gH=cv2.equalizeHist(g)
# rH=cv2.equalizeHist(r)
# result=cv2.merge([bH,gH,rH])    #���� cv2.merge() �� B��G��R ��ͨ���ϲ�Ϊ 3 ͨ�� BGR ��ɫͼ��
# cv2.imshow('dst_rgb',result)






cv2.waitKey(0)