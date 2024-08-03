from keras.layers import Input
from frcnn import FRCNN
from PIL import Image
import numpy as np
import cv2

frcnn = FRCNN()
'''
while (True):
    img = input('img/street.jpg')
    try:
        image = Image.open('img/street.jpg')
    except:
        print('Open Error! Try again!')
        continue

    else:
        r_image = frcnn.detect_image(image)
        print('try')
        r_image.show()
'''

img = cv2.imread('street.jpg')
frame = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 转变成Image
frame = Image.fromarray(np.uint8(frame))

# 进行检测
frame = np.array(frcnn.detect_image(frame))
# RGBtoBGR满足opencv显示格式
frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
cv2.imshow("video",frame)
c= cv2.waitKey()



frcnn.close_session()