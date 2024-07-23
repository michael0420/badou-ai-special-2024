import numpy as np
import utils
import cv2
from tensorflow.keras import backend as K
from model.AlexNet import alexnet
K.set_image_data_format('channels_last')
if __name__ == "__main__":
    model = alexnet()
    model.load_weights("./logs/last1.h5")
    img = cv2.imread("./hgbhwngm.png")
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255
    img_nor = np.expand_dims(img_nor,axis = 0)
    img_resize = utils.resize_image(img_nor,(224,224))
    pre = model.predict(img_resize)
    # print(pre)
    print("标签为%s" %(np.argmax(pre)))
    print("类别为%s" %(utils.print_answer(np.argmax(pre))))
    cv2.imshow("ooo",img)
    cv2.waitKey(0)