import  cv2
import  numpy as np

gray = cv2.imread("../lenna.png", 0)
canny= cv2.Canny(gray,100,200)
cv2.imshow("canny",canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
