import numpy as np
import cv2

img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = img_gray / 255.0
print(img_gray)
img_binary = np.where(img_gray >= 0.5, 1, 0)
print(img_binary)
img_binary = img_binary.astype(np.uint8) * 255
cv2.imshow("binary", img_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()