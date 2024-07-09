import numpy as np
import pic_cut
import cv2
from keras import backend as K
from model.AlexNet import AlexNet

#配置Keras的图像维度顺序为TensorFlow('tf')
K.set_image_data_format('channels_last')

if __name__ == "__main__":
    # 加载模型
    model = AlexNet()
    model.load_weights("./logs/last1.h5")

    # 读取并预处理图像
    img = cv2.imread("Test.jpg")
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img_RGB / 255.0  # 归一化到 [0, 1] 范围
    img_nor = np.expand_dims(img_nor, axis=0)  # 扩展维度以符合模型输入形状

    # 调整图像大小
    img_resize = pic_cut.resize_image(img_nor, (224, 224))

    # 预测并打印结果
    print(pic_cut.print_answer(np.argmax(model.predict(img_resize))))

