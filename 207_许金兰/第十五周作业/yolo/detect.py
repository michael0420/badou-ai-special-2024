import numpy as np
import tensorflow as tf
from yolo_predict import yolo_predictor
from PIL import Image, ImageFont, ImageDraw

def load_weights(var_list, weights_file):
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        if 'conv2d' in var1.name.split('/')[-2]:
            if 'batch_normalization' in var2.name.split('/')[-2]:
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                i += 4
            elif 'conv2d' in var2.name.split('/')[-2]:
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                i += 1
            shape = var1.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


def letterbox_image(image, size):
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
    new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))
    resized_image = image.resize((new_w,new_h), Image.BICUBIC)  # 图像重采样滤波器方法Image.BICUBIC：双立方插值法

    boxed_image = Image.new('RGB', size, (128, 128, 128))   # 创建一张图片Image.new(mode, size, color)
    boxed_image.paste(resized_image, ((w-new_w)//2,(h-new_h)//2))  # 将图片resized_image粘贴到图片boxed_image的((w-new_w)//2,(h-new_h)//2)位置
    return boxed_image

def detect(image_path, model_path, yolo_weights=None,obj_threshold=0.5,nms_threshold=0.5,classes_path='./model_data/coco_classes.txt',anchors_path='./model_data/yolo_anchors.txt'):
    image = Image.open(image_path)
    # 对预测输入图像进行缩放
    resize_image = letterbox_image(image, (416, 416))
    image_data = np.array(resize_image, dtype=np.float32)
    # 归一化
    image_data /= 255.
    # 转格式，填充第一维度
    image_data = np.expand_dims(image_data, axis=0)
    input_image_shape = tf.placeholder(dtype=tf.int32, shape=(2,))
    input_image = tf.placeholder(shape=[None, 416, 416, 3], dtype=tf.float32)
    predictor = yolo_predictor(obj_threshold, nms_threshold, classes_path, anchors_path)
    with tf.Session() as sess:
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
                    input_image: image_data,
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
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = predictor.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
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
        image.save('./img/result1.jpg')

if __name__ == '__main__':
    pre_train_yolo3=True
    image_file="./img/img.jpg"
    model_dir='./test_model/model.ckpt-192192'
    yolo3_weights_path='./model_data/yolov3.weights'

    if pre_train_yolo3 == True:
        detect(image_file, model_dir, yolo3_weights_path)

    else:
        detect(image_file, model_dir)
