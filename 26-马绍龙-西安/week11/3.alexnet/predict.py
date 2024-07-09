import numpy as np
import utils
import cv2
from keras import backend as K
from AlexNet import AlexNet

# K.set_image_dim_ordering('tf')
# K.image_data_format() == 'channels_first'

if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("./last1.h5")
    img = cv2.imread("./Test.jpg")
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img_RGB / 255
    img_nor = np.expand_dims(img_nor, axis=0)  # 在第0个轴上增加一个维度，因为许多模型要求输入数据具有一个batch维度,img_nor.shape[0]==1
    img_resize = utils.resize_image(img_nor, (224, 224))

    print("预测结果为：" + utils.print_answer(np.argmax(model.predict(img_resize))))

    cv2.imshow("img", img)
    cv2.waitKey(0)
