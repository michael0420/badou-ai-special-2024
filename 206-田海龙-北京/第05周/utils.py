
import inspect
import os

import cv2
import numpy as np

# 当前文件路径
current_file_path = inspect.getframeinfo(inspect.currentframe()).filename
# 当前文件所在路径，方便拼接当前路径下文件，不需要从根目录一级一级拼接 
current_directory = os.path.dirname(os.path.abspath(current_file_path))


def cv_imread(file_path,flag=-1):
    """
    读取图像，解决imread不能读取中文路径路径的问题
    :param file_path: 图像路径
    """

    buf=np.fromfile(file_path,dtype=np.uint8)
    
    #imdedcode读取的是RGB图像
    # flag 代表获取的通道数，指定为0，即获取灰度图像
    # flag 指定为1，即获取彩色图像，或cv2.IMREAD_COLOR
    cv_img = cv2.imdecode(buf,flag)

    return cv_img
