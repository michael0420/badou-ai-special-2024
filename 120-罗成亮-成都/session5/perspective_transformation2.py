import cv2
import numpy as np

img = cv2.imread('2.jpg')

result3 = img.copy()

src = np.float32([[0, 0], [492, 0], [0, 1043], [492, 1043]])
dst = np.float32([[140, 753], [632, 147], [1359, 1530], [1655, 1190]])
print(img.shape)

m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (1655, 1530))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
