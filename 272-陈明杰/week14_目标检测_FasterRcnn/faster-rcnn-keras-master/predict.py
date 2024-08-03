from keras.layers import Input
from frcnn import FRCNN
from PIL import Image

frcnn = FRCNN()

if __name__ == '__main__':
    image = Image.open('img/street.jpg')
    r_image = frcnn.detect_image(image)
    r_image.show()
    frcnn.close_session()


# if __name__ == '__main__':
#     while True:
#         img = input('img/street.jpg')
#         try:
#             image = Image.open('img/street.jpg')
#         except:
#             print('Open Error! Try again!')
#             continue
#         else:
#             r_image = frcnn.detect_image(image)
#             r_image.show()
#     frcnn.close_session()
