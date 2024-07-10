import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet
K.set_image_data_format('channels_last')
#K.set_image_dim_ordering('tf')

if __name__ == "__main__":
    #运行模型，载入训练好的数据
    model = AlexNet()
    model.load_weights("./logs/last1.h5")

    img = cv2.imread('./Test.jpg')
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = img_rgb/255
    img_nor = np.expand_dims(img_nor,axis = 0)  #添加0维度，也就是行，也就是在最前面添加维度
    img_resize = utils.resize_image(img_nor,(224,224))

    answer = utils.print_answer(np.argmax(model.predict(img_resize)))
    print('The answer is:',answer)
    print(img_resize.shape[0])
    print(img_resize.shape[1])
    print(img_resize.shape[2])
    cv2.imshow("img", img)
    cv2.waitKey(0)

