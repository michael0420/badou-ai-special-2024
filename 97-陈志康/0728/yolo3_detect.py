import os
import config
import argparse
import numpy as np
import tensorflow as tf
from yolo_predict import yolo_predictor
from PIL import Image, ImageFont, ImageDraw
from utils import letterbox_image, load_weights

# 指定使用GPU的Index
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index


def detect(image_path, model_path, yolo_weights=None):
    """
    Introduction
    ------------
        加载模型，进行预测
    Parameters
    ----------
        model_path: 模型路径，当使用yolo_weights无用
        image_path: 图片路径
    """
    # ---------------------------------------#
    #   图片预处理
    # ---------------------------------------#
    image = Image.open(image_path)
    # 对预测输入图像进行缩放，按照长宽比进行缩放，不足的地方进行填充
    resize_image = letterbox_image(image, (416, 416))
    image_data = np.array(resize_image, dtype=np.float32)
    # 归一化
    image_data /= 255.
    # 转格式，第一维度填充
    image_data = np.expand_dims(image_data, axis=0)
    # ---------------------------------------#
    #   图片输入
    # ---------------------------------------#
    # input_image_shape原图的size
    input_image_shape = tf.compat.v1.placeholder(dtype=tf.int32, shape=(2,))
    # 图像
    input_image = tf.compat.v1.placeholder(shape=[None, 416, 416, 3], dtype=tf.float32)

    # 进入yolo_predictor进行预测，yolo_predictor是用于预测的一个对象
    predictor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path)
    with tf.Session() as sess:
        # ---------------------------------------#
        #   图片预测
        # ---------------------------------------#
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            # 载入模型
            load_op = load_weights(tf.global_variables(scope='predict'), weights_file=yolo_weights)
            sess.run(load_op)

            # 进行预测
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    # image_data这个resize过
                    input_image: image_data,
                    # 以y、x的方式传入
                    input_image_shape: [image.size[1], image.size[0]]
                })
        else:
            boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    input_image: image_data,
                    input_image_shape: [image.size[1], image.size[0]]
                })

        # ---------------------------------------#
        #   画框
        # ---------------------------------------#
        # 找到几个box，打印
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        # 厚度
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            # 获得预测名字，box和分数
            predicted_class = predictor.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            # 打印
            label = '{} {:.2f}'.format(predicted_class, score)

            # 用于画框框和文字
            draw = ImageDraw.Draw(image)
            # textsize用于获得写字的时候，按照这个字体，要多大的框
            label_size = draw.textsize(label, font)

            # 获得四个边
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1] - 1, np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0] - 1, np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            print(label_size)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=predictor.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=predictor.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        image.show()
        image.save('./img/result1.jpg')


if __name__ == '__main__':

    # 当使用yolo3训练好的weights的时候
    if config.pre_train_yolo3 == True:
        detect(config.image_file, config.model_dir, config.yolo3_weights_path)

    # 当使用自训练模型的时候
    else:
        detect(config.image_file, config.model_dir)
