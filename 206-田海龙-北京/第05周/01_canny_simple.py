
import cv2
from utils import current_directory,cv_imread

img_path=current_directory+"\\img\\lenna.png"

def to_canny():
    img=cv_imread(img_path)
    # 灰度化，然后直接调用cv2的canny方法
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_canny=cv2.Canny(img_gray,100,200)
    cv2.imshow("img_canny",img_canny)

    cv2.waitKey()

to_canny()