import numpy as np
import utils
import cv2
from model.AlexNet import AlexNet

if __name__ =='__main__':
    model = AlexNet()
    model.load_weights('last1.h5')
    img = cv2.imread('Test.jpg')
    img_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img_rbg / 255
    #维度扩展
    img_nor = np.expand_dims(img_nor, axis=0)
    img_resize = utils.resize_image(img_nor, (224, 224))
    lable_value = np.argmax( model.predict(img_resize))
    utils.print_answer(lable_value)
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()