import os
import config
import cv2
import numpy as np
import tensorflow as tf
from yolo_predict import yolo_predictor
from PIL import Image, ImageFont, ImageDraw
from utils import letterbox_image, load_weights

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index  # 指定使用GPU的Index

def detect_from_stream(model_path, yolo_weights=None):
    """
    Introduction
    ------------
        加载模型，进行实时摄像头检测
    Parameters
    ----------
        model_path: 模型路径，当使用yolo_weights无用
    """
    # 初始化摄像头
    cap = cv2.VideoCapture(0)  # 使用默认摄像头

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # 定义TensorFlow占位符
    input_image_shape = tf.placeholder(dtype=tf.int32, shape=(2,))
    input_image = tf.placeholder(shape=[None, 416, 416, 3], dtype=tf.float32)

    # 进入yolo_predictor进行预测，yolo_predictor是用于预测的一个对象
    predictor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path)
    with tf.Session() as sess:
        # 加载模型
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            load_op = load_weights(tf.global_variables(scope='predict'), weights_file=yolo_weights)
            sess.run(load_op)
        else:
            boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            saver = tf.train.Saver()
            saver.restore(sess, model_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            resize_image = letterbox_image(image, (416, 416))
            image_data = np.array(resize_image, dtype=np.float32) / 255.
            image_data = np.expand_dims(image_data, axis=0)

            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    input_image: image_data,
                    input_image_shape: [image.size[1], image.size[0]]
                })

            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
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
                bottom = min(image.size[1]-1, np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0]-1, np.floor(right + 0.5).astype('int32'))
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

            result = np.asarray(image)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imshow("YOLO Object Detection", result)
            # 如果窗口被关闭，跳出循环
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("YOLO Object Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # 当使用yolo3训练好的weights的时候
    if config.pre_train_yolo3 == True:
        detect_from_stream(config.model_dir, config.yolo3_weights_path)
    # 当使用自训练模型的时候
    else:
        detect_from_stream(config.model_dir)
