'''
读取本地模型，进行推理
'''
from model.AlexNet import AlexNet
import cv2
import numpy as np
import utils
from tensorflow.keras import backend as K

K.image_data_format() == 'channels_first'

if __name__ == '__main__':

    model = AlexNet()
    dir_model = ['./log/train_model.h5', 'last1.h5']
    model.load_weights(dir_model[0])
    img = cv2.imread('Test3.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img / 255.0
    img_nor = np.expand_dims(img_nor, axis=0)
    img_resize = utils.resize_image(img_nor, (224, 224))
    answer = np.argmax(model.predict(img_resize))
    print(utils.print_answer(answer))
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyWindow('img')
