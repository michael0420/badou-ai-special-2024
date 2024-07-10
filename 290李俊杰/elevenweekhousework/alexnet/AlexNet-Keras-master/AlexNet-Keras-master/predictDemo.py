import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNetDemo import AlexNet


K.image_data_format()=='channels_first'

if __name__ == '__main__':
    model=AlexNet()
    model.load_weights("./logs/")
    image=cv2.imread("./Test.jpg")
    image_RGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img_nor=image_RGB/255
    img_nor=np.expand_dims(img_nor,axis=0)
    img_resize=utils.resize_image(img_nor,(224,224))

    print(utils.print_answer(np.argmax(model.predict(img_resize))))

    cv2.imshow("iii",image)
    cv2.waitKey(0)