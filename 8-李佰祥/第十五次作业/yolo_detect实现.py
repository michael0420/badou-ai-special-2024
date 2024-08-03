import os
import config
import argparse
import numpy as np
import tensorflow as tf

from CV.深度学习.第十五次课yoloV3和人脸识别网络.yolo_predictor_Myclass import yolo_predictor_Myclass
from yolo_predict import yolo_predictor
from PIL import Image, ImageFont, ImageDraw
from utils import load_weights

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index

def letterbox_image(image,size):
    #这种缩放需要保证图片目标的长宽比不变
    w,h = image.size
    d_w,d_h = size
    new_h = int(h * min(d_w/w,d_h/h))
    new_w = int(w * min(d_w/w,d_h/h))
    resized_image = image.resize((new_w,new_h),Image.BICUBIC)
    boxed_image = Image.new('RGB',size,(128,128,128))
    #((w-new_w)//2,(h-new_h)//2) 是一个元组，
    # 表示缩放后的图像 resized_image 在 boxed_image 中的左上角坐标
    #这个坐标确保了 resized_image 在 boxed_image 中居中
    boxed_image.paste(resized_image,((d_w - new_w)//2,(d_h - new_h)//2))
    return boxed_image






def detect(img_path,model_path,yolo_weight=None):
    image = Image.open(img_path)
    resize_image = letterbox_image(image,(416,416))
    image_data = np.array(resize_image,dtype=np.float32)
    #归一化
    image_data = image_data/255
    image_data = np.expand_dims(image_data,0)

    input_image_shape= tf.placeholder(tf.int32,[2,])
    input_image = tf.placeholder(tf.float32,[None,416,416,3])

    #调用yolo_predict
    predictor = yolo_predictor_Myclass(config.obj_threshold,config.nms_threshold,config.classes_path,config.anchors_path)
    with tf.Session() as sess:
        if yolo_weight is not None:
            print('进入if')
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_image,input_image_shape)
                load_op = load_weights(tf.global_variables(scope='predict'),weights_file=yolo_weight)
                sess.run(load_op)

                #预测
                out_boxes,out_scores,out_classes = sess.run(
                    [boxes,scores,classes],
                    feed_dict={input_image: image_data,
                               input_image_shape:[image.size[1],image.size[0]]
                            }
                )
        else:
            boxes,scores,classes = predictor.predict(input_image,input_image_shape)
            saver = tf.train.Saver()
            saver.restore(sess,model_path)
            out_boxes,out_scores,out_classes = sess.run(
                [boxes,scores,classes],
                feed_dict={input_image: image_data,
                           input_image_shape:[image.size[1],image.size[0]]
                           }
            )
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
        image.save('./img/result2.jpg')


























if __name__ == '__main__':
    if config.pre_train_yolo3 ==True:

        detect(config.image_file,config.model_dir,config.yolo3_weights_path)
    else:
        detect(config.image_file,config.model_dir)



