"""
其他函数工具
"""
import numpy as np , cv2
from modules.global_params import CLASSES

def get_label(y_pred):
    return np.array(CLASSES)[ y_pred.argmax(axis=1)][0]

def display_img_label(img, label):
    cv2.imshow(label, img)
    cv2.waitKey(0)