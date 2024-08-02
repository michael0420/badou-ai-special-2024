import cv2
from mtcnn_homework import mtcnn

img = cv2.imread('img/test1.jpg')

model = mtcnn()
threshold = [0.5, 0.6, 0.7]
rectangles = model.detectFace(img, threshold)
draw = img.copy()

for rectangle in rectangles:
    if rectangle is not None:
        W = -int(rectangle[0]) + int(rectangle[2])
        H = -int(rectangle[1]) + int(rectangle[3])
        paddingH = 0.01 * W
        paddingW = 0.01 * H
        crop_img = img[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
        if crop_img is None:
            continue
        if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
            continue

        cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                      (255, 0, 0), 1)

        for i in range(5, 15, 2):
            cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0))

cv2.imwrite('img/detect_homework.jpg', draw)
cv2.imshow('test', draw)
cv2.waitKey(0)