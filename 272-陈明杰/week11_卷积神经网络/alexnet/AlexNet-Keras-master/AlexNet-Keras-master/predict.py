import numpy as np
import utils
import cv2
from keras import backend as K
from model.AlexNet import AlexNet

# K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_last')


if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("logs/last1.h5")
    img = cv2.imread("./Test.jpg")
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255
    img_nor = np.expand_dims(img_nor,axis = 0)
    img_resize = utils.resize_image(img_nor,(224,224))
    #utils.print_answer(np.argmax(model.predict(img)))
    print(utils.print_answer(np.argmax(model.predict(img_resize))))
    cv2.imshow("ooo",img)
    cv2.waitKey(0)

# # 第一次
# if __name__=='__main__':
#     # 构建模型
#     model=AlexNet()
#     # 读取已经训练好的模型参数
#     model.load_weights('./logs/last1.h5')
#     # 读取需要推理的图片
#     img=cv2.imread('Test.jpg')
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img=img/255
#     img=np.expand_dims(img,axis=0)
#     img=utils.resize_image(img,(24,24))
#     # 进行推理
#     print(utils.print_answer(np.argmax(model.predict(img))))
#     cv2.imshow("img",img)
#     cv2.waitKey(0)
#
# # 第二次
# if __name__=='__main__':
#     # 构建模型
#     model=AlexNet()
#     # 读取已经训练好的模型数据
#     model.load_weights('./xxxxxxxxxxxx')
#     # 读取图片
#     img=cv2.imread("Test.jpg")
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     # 归一化
#     img=img/255
#     # 添加一个维度，因为推理要求是四维的(batch_size,high,weight,channels)
#     img=np.expand_dims(img,axis=0)
#     # resize图片的尺寸，每个网络对图片的尺寸可能有不同的要求
#     img=utils.resize_image(img,(224,224))
#     # 推理
#     print(utils.print_answer(np.argmax(model.predict(img))))
#     cv2.imshow("img",img)
#     cv2.waitKey(0)
#
# # 第三次
# if __name__=='__main__':
#     # 构建模型
#     model=AlexNet()
#     model.load_weights('./xxxxxxxxxxxxxxx')
#     # 读取图片
#     img=cv2.imread("./xxxxxxxxxxxx")
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img=img/255
#     # 添加一个维度
#     img=np.expand_dims(img,axis=0)
#     # 修改图片的大小为224*224
#     img=utils.resize_image(img,(224,224))
#     # 推理
#     print(utils.print_answer(np.argmax(model.predict(img))))
#     cv2.imshow("img",img)
#     cv2.waitKey(0)
#
#
# # 第四次
# if __name__=='__main__':
#     # 构建模型
#     model=AlexNet()
#     # 导入模型参数
#     model.load_weights('./xxxxxxx')
#     # 读取图片并对图片进行一些处理
#     img=cv2.imread('./xxxxxxx')
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img=img/255
#     img=np.expand_dims(img,axis=0)
#     img=utils.resize_image(img,(224,224))
#     print(utils.print_answer(np.argmax(model.predict(img))))
#     cv2.imshow("img",img)
#     cv2.waitKey(0)
#
#
# # 第五次
# if __name__=='__main__':
#     # 构建模型
#     model=AlexNet()
#     model.load_weights('./xxxx')
#     # 读取数据并进行处理
#     img=cv2.imread('./xxxxxxxx')
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     img=img/255
#     # 添加维度
#     img=np.expand_dims(img,axis=0)
#     img=utils.resize_image(img,(224,224))
#     print(utils.print_answer(np.argmax(model.predict(img))))
#     cv2.imshow('img',img)
#     cv2.waitKey(0)
#
# # 第六次
# if __name__=='__main__':
#     # 构建模型
#     model=AlexNet()
#     # 读取模型权重数据
#     model.load_weights('')
#     pass