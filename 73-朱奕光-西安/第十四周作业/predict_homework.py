from frcnn_homework import FRCNN
from PIL import Image

frcnn = FRCNN
img = Image.open('street.jpg')
res = frcnn.detect_image(img)
res.show()
frcnn.close_session()