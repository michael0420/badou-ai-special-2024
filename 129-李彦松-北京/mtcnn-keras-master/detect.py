import cv2
import numpy as np
from mtcnn import mtcnn

img = cv2.imread('img/timg.jpg')

model = mtcnn()
print("P-Net summary:")
model.Pnet.summary()
print("R-Net summary:")
model.Rnet.summary()
print("O-Net summary:")
model.Onet.summary()
threshold = [0.5,0.6,0.7]  # 三段网络的置信度阈值不同
rectangles = model.detectFace(img, threshold)
draw = img.copy()

for rectangle in rectangles:
    if rectangle is not None:
        W = -int(rectangle[0]) + int(rectangle[2])
        H = -int(rectangle[1]) + int(rectangle[3])
        paddingH = 0.01 * W
        paddingW = 0.02 * H
        crop_img = img[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
        if crop_img is None:
            continue
        if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
            continue
        cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)

        for i in range(5, 15, 2):
            cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))

cv2.imwrite("img/out.jpg",draw)

cv2.imshow("test", draw)
c = cv2.waitKey(0)
