import numpy as np
import utils
import cv2
from tensorflow.keras import backend as K
from model.AlexNet import AlexNet

# K.set_image_dim_ordering('tf')  # hwc
K.image_data_format() == 'channels_first'

if __name__ == '__main__':
    model = AlexNet()
    model.load_weights('./logs/last1.h5')  # 加载预训练的模型权重
    img = cv2.imread('./Test.jpg')
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图片从BGR格式转换为RGB格式，因为Keras模型通常期望RGB输入
    img_nor = img_RGB/255
    # 增加一个维度，将图片形状从(height, width, channels)变为(1, height, width, channels)，以符合模型输入要求
    img_nor = np.expand_dims(img_nor, axis=0)
    img_resize = utils.resize_image(img_nor, (224, 224))
    print(utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow('image', img)
    cv2.waitKey(0)



