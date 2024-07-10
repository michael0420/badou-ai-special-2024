import numpy as np
import utils
import cv2
from keras import backend as K
from alexnetModel import AlexNet

K.image_data_format() == 'channels_first'

if __name__ == '__main__':
    model = AlexNet()
    model.load_weights('./logs/last1.h5')
    img = cv2.imread('./Test.jpg')
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img_normlization = img_rgb / 255
    img_normlization = np.expand_dims(img_normlization,axis=0)

    img_resize = utils.resize_img(img_normlization,(224,224))

    print(utils.print_answer(np.argmax(model.predict(img_resize))))

    cv2.imshow('frist_test',img)
    cv2.waitKey(0)


