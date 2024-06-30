import cv2
import numpy as np

from __init__ import cv_imread, current_directory
import os


def combine_img():
    """
    图像拼接
    """
    
    img_path1 = os.path.join(current_directory, "img", "iphone1.png")
    img_path2 = os.path.join(current_directory, "img", "iphone2.png")
    img1 = cv_imread(img_path1)
    img2 = cv_imread(img_path2)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    print(h1, w1, h2, w2)

    # img1 = cv2.resize(img1, (w2, h2))
    # img = np.hstack((img1, img2))
    # cv2.imshow("combine", img)

    des = np.zeros((max(h1, h2),w1 + w2,  3), np.uint8)
    des[:h1, :w1] = img1
    des[:h2, w1:] = img2
    cv2.imshow("combine", des)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


combine_img()
