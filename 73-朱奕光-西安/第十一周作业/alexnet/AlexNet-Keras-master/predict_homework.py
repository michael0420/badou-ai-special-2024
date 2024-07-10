import keras
from AlexNet_Keras_homework import AlexNet
from keras import backend as K
import cv2
from utils_homework import img_resize
from utils_homework import answer
import numpy as np

K.set_image_data_format('channels_last')

model = AlexNet()
model.load_weights('./logs/朱奕光last1.h5')   #出现AttributeError: 'str' object has no attribute 'decode'报错，降级H5PY至2.x版本解决
img = cv2.imread('Test.jpg')
img = img/255
img = img_resize(img, (224,224))
res = answer(np.argmax(model.predict(img)))
print(res)
cv2.imshow('test',img)
cv2.waitKey(0)