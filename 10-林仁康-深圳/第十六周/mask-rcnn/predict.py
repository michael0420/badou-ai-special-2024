from keras.layers import Input
from mask_rcnn import MASK_RCNN 
from PIL import Image

mask_rcnn = MASK_RCNN()

while True:
    img = input('img/street.jpg')
    try:
        image = Image.open('img/street.jpg')
    except:
        print('Open Error! Try again!')
        continue
    else:
        mask_rcnn.detect_image(image)
mask_rcnn.close_session()
    
