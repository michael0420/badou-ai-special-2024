
import cv2
import numpy as np


def cv_imread(file_path):
    """
    读取图像，解决imread不能读取中文路径路径的问题
    :param file_path: 图像路径
    """

    buf=np.fromfile(file_path,dtype=np.uint8)
    
    #imdedcode读取的是RGB图像
    cv_img = cv2.imdecode(buf,-1)

    return cv_img

