import numpy as np
import cv2

# 实现一维线性插值
# 已知x坐标求y
def linear_interp(x, x1, x2, y1, y2):
    '''
    Parameters:
    - x: a float value representing the x-coordinate of the target point
    - x1, x2: x-coordinates of the two known points
    - y1, y2: y-coordinates (values) of the two known points
    Returns:
    - y: the interpolated value at the target point
    '''
    y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    return y

# 已知y坐标求x
def inverse_linear_interp(y, x1, x2, y1, y2):
    '''
    Parameters:
    - y: a float value representing the y-coordinate (value) of the target point
    - x1, x2: x-coordinates of the two known points
    - y1, y2: y-coordinates (values) of the two known points
    Returns:
    - x: the interpolated x-coordinate at the target point
    '''
    x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return y