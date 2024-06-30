#需安装skimage库

import cv2
from skimage.color import rgb2gray
img = cv2.imread(r'E:\AI\CV\second week\work\lenna.png')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',img_gray)
