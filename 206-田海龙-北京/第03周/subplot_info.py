
import matplotlib.pyplot as plt
import numpy as np
from Util import cv_imread

class SubplotInfo:
    """
    子图设置信息类
    """

    def __init__(self,img_row,img_col):
        """
        初始化
        :param img_row: 图像行数
        :param img_col: 图像列数
        """
        # todo：img_row、img_col参数验证
        self.img_row = img_row
        self.img_col = img_col
        self.cur_index = 0

    def set_subplot(self,img_index=None):
        """
        设置子图
        """    

        # 如果参数未设置子图位置索引
        if img_index == None:
            # 如果当前子图索引有值
            if self.cur_index != None:
                # 直接+1，幅值
                self.cur_index += 1   
                img_index=self.cur_index
            else:
                self.cur_index = 1   
                img_index=self.cur_index
        
        plt.subplot(self.img_row,self.img_col,img_index)

        self.cur_index =img_index
