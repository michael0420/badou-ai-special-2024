
import cv2
import numpy as np

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

import win32gui
# import win32con

def cv_set_titile(oldTitle,newTitle='中文',oneRun=False):
    """
    设置窗口标题
    :param oldTitle: 旧标题
    :param newTitle: 新标题
    :param oneRun: 是否只运行一次
    :return:
    """
    if oneRun == False:
        # 根据窗口名称查找其句柄 然后使用函数修改其标题
        # 尽量选择一个不常见的英文名 防止误该已有#的窗口标题 初始化时通常取无意义的名字  比如这里取‘aa’
        handle = win32gui.FindWindow(0, oldTitle)
        win32gui.SetWindowText(handle, newTitle)
        oneRun= True
    return oneRun



