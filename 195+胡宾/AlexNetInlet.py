import numpy as np
import myutils
import cv2
from computervision.alexnet.myAlexNet import my_AlexNet

# K.set_image_dim_ordering('tf')

if __name__ == "__main__":
    model = my_AlexNet()
    model.load_weights("./logs/ep039-loss0.004-val_loss0.652.h5")
    img = cv2.imread("./Test.jpg")
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255
    img_nor = np.expand_dims(img_nor,axis = 0)
    img_resize = myutils.resize_image(img_nor,(224,224))
    #utils.print_answer(np.argmax(model.predict(img)))
    print(myutils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow("ooo",img)
    cv2.waitKey(0)