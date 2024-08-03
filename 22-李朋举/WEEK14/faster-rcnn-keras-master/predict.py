from keras.layers import Input
from frcnn import FRCNN 
from PIL import Image

# ---------------------------------------------------#
#   Detect（导入模型 + 检测图片）
# ---------------------------------------------------#

# 导入模型
frcnn = FRCNN()

while True:
    img = input('img/street.jpg')
    try:
        image = Image.open('img/street.jpg')
    except:
        print('Open Error! Try again!')
        continue
    else:
        # 检测图片
        r_image = frcnn.detect_image(image)
        r_image.show()
frcnn.close_session()
    
