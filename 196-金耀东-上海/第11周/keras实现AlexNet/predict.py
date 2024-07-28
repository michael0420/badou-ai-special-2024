from modules import models , dataset , utils
from modules.global_params import PATH_LAST_MODEL, IMG_SHAPE, CLASSES, PATH_DATA_DIR
from keras.preprocessing.image import load_img , img_to_array
import numpy as np
import cv2
import os

if __name__ == "__main__":

    # 测试图片路径
    test_img_path = os.path.join(PATH_DATA_DIR, "sample.jpg")

    # 创建模型
    model = models.AlexNet_BN()

    # 加载训练完毕的权重
    model.load_weights(PATH_LAST_MODEL)

    # 加载测试图片,并归一化
    img_test = img_to_array( load_img(test_img_path, target_size=IMG_SHAPE[:2]) ) / 255.0
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

    # # 拓展维度
    x_test = np.expand_dims( img_test, axis=0 )

    # 进行预测
    y_pred = model.predict(x_test)

    # 获取预测结果
    label = utils.get_label(y_pred)
    print(f"predict:{label}")

    # 展示预测结果,图片标题即为预测结果
    utils.display_img_label(img_test, label)