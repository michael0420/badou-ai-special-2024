from __future__ import division
from nets.frcnn import get_model
from nets.frcnn_training import cls_loss, smooth_l1, Generator, get_img_output_length, class_loss_cls, class_loss_regr

from utils.config import Config # 用于加载配置参数
from utils.utils import BBoxUtility
from utils.roi_helpers import calc_iou

from keras.utils import generic_utils
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import keras
import numpy as np
import time
import tensorflow as tf
from utils.anchors import get_anchors


def write_log(callback, names, logs, batch_no):  #将训练过程中的日志写入到 TensorBoard。它接受回调函数、日志名称和日志内容作为参数。
    '''
   该函数将日志信息写入到 TensorBoard。
   参数 callback 是回调函数，names 是日志名称数组，logs 是日志内容数组，batch_no 是批次编号。
    '''

    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


if __name__ == "__main__":
    config = Config()
    NUM_CLASSES = 21
    EPOCH = 100
    EPOCH_LENGTH = 2000
    # 初始化 BBoxUtility 对象，用于处理边界框相关操作，如计算 IoU。
    # 设置标注文件路径为 annotation_path。
    bbox_util = BBoxUtility(overlap_threshold=config.rpn_max_overlap, ignore_threshold=config.rpn_min_overlap)
    annotation_path = '2007_train.txt'

    # 调用 get_model 函数获取三个模型：model_rpn（RPN 模型）、model_classifier（分类器模型）和 model_all（全部模型），并加载预训练权重。
    model_rpn, model_classifier, model_all = get_model(config, NUM_CLASSES)
    base_net_weights = "model_data/voc_weights.h5"

    # 加载预训练权重到 RPN 模型和分类器模型中，by_name=True 表示根据权重文件中参数名加载。
    model_all.summary()
    model_rpn.load_weights(base_net_weights, by_name=True)
    model_classifier.load_weights(base_net_weights, by_name=True)

    # 打开标注文件，将内容读入内存，并随机打乱顺序。
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 创建数据生成器 gen，它将标注数据和配置信息转化为可用于训练的数据。
    # 使用数据生成器创建训练数据迭代器 rpn_train。
    # 设置日志目录为 logs。
    gen = Generator(bbox_util, lines, NUM_CLASSES, solid=True)
    rpn_train = gen.generate()
    log_dir = "logs"

    # 训练参数设置
    # 创建 TensorBoard 回调函数，指定日志目录，并将其赋值给 callback。
    # 将全部模型赋值给回调函数的模型参数，以便 TensorBoard 日志记录。

    #logging = TensorBoard(log_dir=log_dir)
    logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    callback = logging
    callback.set_model(model_all)

    '''
    编译 RPN 模型，使用 smooth_l1 损失函数进行回归，使用 cls_loss 进行分类，优化器为 Adam，初始学习率为 1e-5。
    编译分类器模型，使用 class_loss_cls 和 class_loss_regr 作为损失函数，准确率作为指标，优化器同样为 Adam，初始学习率为 1e-5。
    编译全部模型，使用 SGD 优化器，损失函数为平均绝对误差（mae）。
    '''
    model_rpn.compile(loss={
        'regression': smooth_l1(),
        'classification': cls_loss()
    }, optimizer=keras.optimizers.Adam(lr=1e-5)
    )
    model_classifier.compile(loss=[
        class_loss_cls,
        class_loss_regr(NUM_CLASSES - 1)
    ],
        metrics={'dense_class_{}'.format(NUM_CLASSES): 'accuracy'}, optimizer=keras.optimizers.Adam(lr=1e-5)
    )
    model_all.compile(optimizer='sgd', loss='mae')

    # 初始化参数
    iter_num = 0
    train_step = 0
    losses = np.zeros((EPOCH_LENGTH, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()
    # 最佳loss
    best_loss = np.Inf
    # 数字到类的映射
    print('Starting training')

    for i in range(EPOCH):

        if i == 20:
            model_rpn.compile(loss={
                'regression': smooth_l1(),
                'classification': cls_loss()
            }, optimizer=keras.optimizers.Adam(lr=1e-6)
            )
            model_classifier.compile(loss=[
                class_loss_cls,
                class_loss_regr(NUM_CLASSES - 1)
            ],
                metrics={'dense_class_{}'.format(NUM_CLASSES): 'accuracy'}, optimizer=keras.optimizers.Adam(lr=1e-6)
            )
            print("Learning rate decrease")

        progbar = generic_utils.Progbar(EPOCH_LENGTH)
        print('Epoch {}/{}'.format(i + 1, EPOCH))
        while True:
            if len(rpn_accuracy_rpn_monitor) == EPOCH_LENGTH and config.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                    mean_overlapping_bboxes, EPOCH_LENGTH))
                if mean_overlapping_bboxes == 0:
                    print(
                        'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            X, Y, boxes = next(rpn_train)  # 从训练数据迭代器中获取下一批数据，X 为输入图像数据，Y 为相应的标签，boxes 为真实边界框。

            loss_rpn = model_rpn.train_on_batch(X, Y)
            write_log(callback, ['rpn_cls_loss', 'rpn_reg_loss'], loss_rpn, train_step)
            P_rpn = model_rpn.predict_on_batch(X)
            height, width, _ = np.shape(X[0])
            anchors = get_anchors(get_img_output_length(width, height), width, height) #get_anchors 函数获取锚点（anchors），锚点根据图像尺寸计算

            # 将预测结果进行解码
            # 使用 BBoxUtility 将 RPN 的预测输出解码为边界框，results 是解码后的边界框信息，R 是边界框的坐标和分数。
            results = bbox_util.detection_out(P_rpn, anchors, 1, confidence_threshold=0)

            R = results[0][:, 2:]

            # 计算 RPN 预测的边界框和真实边界框的 IoU，并获取相关指标。
            X2, Y1, Y2, IouS = calc_iou(R, config, boxes[0], width, height, NUM_CLASSES)

            if X2 is None:   # 如果无法计算 IoU，则将 0 加入相关数组，并跳过本次迭代。
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            # 获取负样本和正样本的索引，Y1 是分类器输出，最后一栏为 1 表示负样本，为 0 表示正样本。
            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))


            # 如果正样本数量少于一半的 ROI，则选择所有正样本，否则随机选择一半正样本。
            # 尝试从负样本中随机选择与正样本数量相等的负样本，如果负样本不足，则允许重复选择。
            if len(neg_samples) == 0:
                continue

            if len(pos_samples) < config.num_rois // 2:
                selected_pos_samples = pos_samples.tolist()
            else:
                selected_pos_samples = np.random.choice(pos_samples, config.num_rois // 2, replace=False).tolist()
            try:
                selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples),
                                                        replace=False).tolist()
            except:
                selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples),
                                                        replace=True).tolist()

            sel_samples = selected_pos_samples + selected_neg_samples
            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                         [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            write_log(callback, ['detection_cls_loss', 'detection_reg_loss', 'detection_acc'], loss_class, train_step)

            # 更新训练步数和迭代器编号，并使用当前信息更新进度条。
            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]
            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]


            # 更新训练步数和迭代器编号，并使用当前信息更新进度条。
            train_step += 1
            iter_num += 1
            progbar.update(iter_num,
                           [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                            ('detector_cls', np.mean(losses[:iter_num, 2])),
                            ('detector_regr', np.mean(losses[:iter_num, 3]))])

            # 计算本 epoch 的平均损失和准确率，以及 RPN 平均重叠边界框数量。

            if iter_num == EPOCH_LENGTH:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if config.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))


                # 计算当前总损失，重置迭代器编号
                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                write_log(callback,
                          ['Elapsed_time', 'mean_overlapping_bboxes', 'mean_rpn_cls_loss', 'mean_rpn_reg_loss',
                           'mean_detection_cls_loss', 'mean_detection_reg_loss', 'mean_detection_acc', 'total_loss'],
                          [time.time() - start_time, mean_overlapping_bboxes, loss_rpn_cls, loss_rpn_regr,
                           loss_class_cls, loss_class_regr, class_acc, curr_loss], i)

                # 如果当前 epoch 的损失值比之前任何一个 epoch 都低，就会保存该模型的权重，以便后续使用或进一步优化
                if config.verbose:
                    print('The best loss is {}. The current loss is {}. Saving weights'.format(best_loss, curr_loss))
                if curr_loss < best_loss:
                    best_loss = curr_loss
                model_all.save_weights(log_dir + "/epoch{:03d}-loss{:.3f}-rpn{:.3f}-roi{:.3f}".format(i, curr_loss,
                                                                                                      loss_rpn_cls + loss_rpn_regr,
                                                                                                      loss_class_cls + loss_class_regr) + ".h5")

                break