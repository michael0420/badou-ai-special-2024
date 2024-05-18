import cv2
import numpy as np

img = cv2.imread('0448dcf9b53f127c4d085e4dbee263a.jpg')

src = np.float32([[666, 153], [1442, 185], [263, 1203], [1756, 1159]])
dst = np.float32([[0, 0], [337 * 2, 0], [0, 488 * 2], [337 * 2, 488 * 2]])

m = cv2.getPerspectiveTransform(src, dst)

result = cv2.warpPerspective(img.copy(), m, (337 * 2, 488 * 2))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
