import cv2
import numpy as np

img = cv2.imread('default.jpg')

result = img
src = np.float32([[30, 390], [300, 283], [266, 900], [500, 740]])
dst = np.float32([[0, 0], [270, 0], [0, 510], [270, 510]])
a = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(result, a, (270, 510))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
