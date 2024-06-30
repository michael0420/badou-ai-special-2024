
import cv2


def sift(img):
    """
    关键点提取
    """
    # cv2.imshow('original', img)
    # 不需要灰度化
    gray = img

    # 如下方式已警告弃用
    # sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.SIFT_create()
    sift = cv2.SIFT.create()

    # 结果：关键点信息、关键点的详细描述
    keypoints, descriptor = sift.detectAndCompute(gray, None)
    # print(keypoints[0])
    # print(len(keypoints))
    # 128维
    # print(len(descriptor[0]))

    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS对图像的每个关键点都绘制了圆圈和方向。
    img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            color=(51, 163, 236))
                            
    #img=cv2.drawKeypoints(gray,keypoints,img)

    cv2.imshow('sift_keypoints', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import os
from __init__ import current_directory,cv_imread

def sift_test():
    img_path = os.path.join(current_directory, "img", "lenna.png")
    img = cv_imread(img_path)
    sift(img)

sift_test()


    