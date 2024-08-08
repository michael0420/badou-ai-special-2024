import cv2
import numpy as np
from mtcnn_demo import mtcnn

img = cv2.imread('img/timg.jpg')
model = mtcnn()
threshold = [0.5,0.6,0.7]

rectangles = model.detectFace(img,threshold)
draw = img.copy()

for rectangle in rectangles:
    if rectangle is not None:
        # 计算出宽高
        w = -int(rectangle[0]) + int(rectangle[2])
        h = -int(rectangle[1]) + int(rectangle[3])
        paddingH = 0.01 * w
        paddingW = 0.02 * h
        crop_img = img[int(rectangle[1] + paddingH) : int(rectangle[3] - paddingH),int(rectangle[0] - paddingW) : int(rectangle[2] + paddingW)]
        if crop_img is None:
            continue
        if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
            continue
        # 绘制矩形框
        cv2.rectangle(draw,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
        for i in range(5,15,2):
            # 绘制关键点
            cv2.circle(draw,(int(rectangle[0+i]),int(rectangle[1+i])),i,(0,255,0),2)
        cv2.imwrite('img/timg_res.jpg',draw)

        cv2.imshow('test',draw)
        c = cv2.waitKey(0)
