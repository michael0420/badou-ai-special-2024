import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet

K.image_data_format() == 'channels_first'

model = AlexNet()
model.load_weights('./logs/last1.h5')
img = cv2.imread("./Test.jpg")
testImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_nor = img/255
img_nor = np.expand_dims(img_nor, axis=0)
img_resize = utils.resize_image(img_nor, (224, 224))
print(utils.print_answer(np.argmax(model.predict(img_resize))))
cv2.imshow("ooo",img)
cv2.waitKey(0)