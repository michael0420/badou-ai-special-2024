from keras.preprocessing.image import load_img, img_to_array
from os.path import join as path_join
from modules.global_params import PATH_DATA_DIR, PATH_LAST_MODEL, PATH_SYNSET, IMG_SHAPE
from modules import models, utils
from tensorflow.nn import softmax
from tensorflow.train import Saver
import tensorflow.compat.v1 as tf
import numpy as np

if __name__ == "__main__":
    # 加载测试数据
    img_dog = img_to_array( load_img(path_join(PATH_DATA_DIR, "dog.jpg"), target_size=IMG_SHAPE[:2]) )
    img_table = img_to_array( load_img(path_join(PATH_DATA_DIR, "table.jpg"), target_size=IMG_SHAPE[:2]) )
    imgs_test = np.stack([img_dog, img_table])

    # 创建placehold用于存储输入
    x_place = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])

    # 构建神经网络（图）
    y_hat = models.VGG16(inputs=x_place, is_training=False)

    with tf.Session() as sess:
        # 全局参数初始化
        sess.run( tf.global_variables_initializer() )

        # 加载训练完毕的模型
        saver = Saver()
        saver.restore( sess, PATH_LAST_MODEL )

        # 进行预测
        y_pred = sess.run( softmax(y_hat), feed_dict={x_place:imgs_test} )

    # 打印预测结果
    utils.print_top5_predictions(y_pred, PATH_SYNSET)
