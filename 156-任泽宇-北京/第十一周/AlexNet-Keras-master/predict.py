import numpy as np
import utils
import cv2
from keras import backend as K
from AlexNet import AlexNet

# K.set_image_dim_ordering('tf')
K.image_data_format() == 'channels_first'


if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("../../../../../data/AlexNet-Keras-master/last1.h5", by_name=True)
    img = cv2.imread("../../../../../data/AlexNet-Keras-master/test1.png")
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255
    img_nor = np.expand_dims(img_nor, axis=0)
    img_resize = utils.resize_image(img_nor, (224, 224))
    print('the answer is：', utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow("类别", img)
    cv2.waitKey(0)