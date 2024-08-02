from fasterrcnn import FASTERRCNN
from PIL import Image

frcnn = FASTERRCNN()
while True:
    img = input('img/street.jpg')
    try:
        image = Image.open('img/street.jpg')
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = frcnn.detect_image(image)
        r_image.show()
frcnn.close_session()
