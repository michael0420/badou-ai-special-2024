
"""
透视变换
"""

import  cv2
import  numpy as np

img = cv2.imread("../photo1.jpg")
img_c=img.copy()

#实现透视变换
def perspectiveTransform(img_c):
    # h,w = img_c.shape[:2]
    # src = np.float32([[0,0],[w,0],[0,h],[w,h]])
    # dst = np.float32([[0,0],[w,0],[0,h],[w,h]])
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    M = cv2.getPerspectiveTransform(src,dst)
    # result = cv2.warpPerspective(img_c,M,(w,h))
    result = cv2.warpPerspective(img_c,M,(377,488))
    return result

result=perspectiveTransform(img_c)
cv2.imshow("img",img)
cv2.imshow("result",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
