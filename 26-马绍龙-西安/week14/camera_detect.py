from keras.layers import Input
from frcnn import FRCNN
from PIL import Image
import numpy as np
import cv2

frcnn = FRCNN()

# 初始化调用摄像头
capture = cv2.VideoCapture(0)

# 开始一个无限循环，直到用户退出
while (True):
    ref, frame = capture.read()  # 从摄像头读取一帧视频

    # 将BGR图像转换为RGB格式，以适应后续处理
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 将numpy数组转换为PIL Image对象
    frame = Image.fromarray(np.uint8(frame))

    # 使用Faster R-CNN模型对图像进行检测
    frame = np.array(frcnn.detect_image(frame))

    # 将RGB图像转换回BGR格式，以供OpenCV显示
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 显示处理后的视频帧
    cv2.imshow("video", frame)

    # 等待用户按键，如果按下ESC键，则退出循环
    c = cv2.waitKey(30) & 0xff
    if c == 27:
        # 释放摄像头资源
        capture.release()
        break

# 关闭Faster R-CNN的会话
frcnn.close_session()

