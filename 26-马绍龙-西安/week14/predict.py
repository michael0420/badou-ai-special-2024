from keras.layers import Input
from frcnn import FRCNN 
from PIL import Image

# 初始化Faster R-CNN对象，用于后续的图像检测
frcnn = FRCNN()

# 尝试打开指定的图像文件
img = input('img/street.jpg')
try:
    # 使用PIL库打开图像文件
    image = Image.open('img/street.jpg')
except:
    # 如果打开失败，输出错误信息并提示重新尝试
    print('Open Error! Try again!')
else:
    # 使用frcnn对象对图像进行检测，返回检测后的图像
    r_image = frcnn.detect_image(image)
    # 显示检测后的图像
    r_image.show()
# 关闭Faster R-CNN会话，释放资源
frcnn.close_session()

