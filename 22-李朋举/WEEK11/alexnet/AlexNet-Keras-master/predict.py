import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet

# K.set_image_dim_ordering('tf')
K.image_data_format() == 'chnanels_first'

if __name__ == "__main__":
    # 加载预训练的 AlexNet 模型
    model = AlexNet()
    # 加载训练完成后存储的权重
    model.load_weights("./logs/last1.h5")
    # 读取了一张图片，并将其转换为 RGB 格式
    img = cv2.imread("./Test.jpg")
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # 对图片进行了归一化处理，并将其扩展为一个四维张量。
    '''
    `np.expand_dims` 是 NumPy 库中的一个函数，用于在数组的指定位置添加一个新的维度。
        在这个例子中，`np.expand_dims(img_nor, axis=0)` 的作用是在数组 `img_nor` 的第一个位置添加一个新的维度，将其从一个二维数组转换为一个三维数组。
        具体来说，`img_nor` 是一个二维数组，它的形状可能是 `(height, width)`。通过调用 `np.expand_dims(img_nor, axis=0)`，在第一个位置添加了一个新的维度，
        使得数组的形状变为 `(1, height, width)`。这样做的目的是为了与 AlexNet 模型的输入要求相匹配，
        因为 AlexNet 模型通常要求输入的图像是一个四维张量，其中第一个维度表示图像的数量。
    '''
    img_nor = img_RGB/255
    img_nor = np.expand_dims(img_nor,axis = 0)
    # 使用resize_image函数对图片进行了大小调整，使其符合模型的输入要求
    img_resize = utils.resize_image(img_nor,(224,224))
    #utils.print_answer(np.argmax(model.predict(img)))
    # 代码使用模型对调整大小后的图片进行了预测，并打印出预测结果。预测结果是一个整数->转换成中文，表示图片所属的类别。
    print(utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow("ooo",img)
    cv2.waitKey(0)