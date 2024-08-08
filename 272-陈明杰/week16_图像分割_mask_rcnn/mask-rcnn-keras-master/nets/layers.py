import tensorflow as tf
from keras.engine import Layer
import numpy as np
from utils import utils

#----------------------------------------------------------#
#   Proposal Layer
#   该部分代码用于将先验框转化成建议框
#----------------------------------------------------------#

def apply_box_deltas_graph(boxes, deltas):
    # 计算先验框的中心和宽高
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width

    # height=boxes[:,2]-boxes[:,0]
    # width=boxes[:,3]-boxes[:,1]
    # center_y=boxes[:,0]+0.5*height
    # center_x=boxes[:,1]+0.5*width

    # 计算出调整后的先验框的中心和宽高
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])

    # center_y+=deltas[:,0]*height
    # center_x=deltas[:,1]*width
    # height*=tf.exp(deltas[:,2])
    # width*=tf.exp(deltas[:,3])

    # 计算左上角和右下角的点的坐标
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")

    # y1=center_y-0.5*height
    # x1=center_x-0.5*width
    # y2=center_y+0.5*height
    # x2=center_x+0.5*width
    # result=tf.stack([y1,x1,y2,x2],axis=1,name='apply_box_deltas_out')

    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped

class ProposalLayer(Layer):

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
    # [rpn_class, rpn_bbox, anchors]
    def call(self, inputs):

        # 代表这个先验框内部是否有物体[batch, num_rois, 1]
        scores = inputs[0][:, :, 1]

        # 代表这个先验框的调整参数[batch, num_rois, 4]
        deltas = inputs[1]

        # [0.1 0.1 0.2 0.2]，改变数量级
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])

        # Anchors
        anchors = inputs[2]

        # 筛选出得分前6000个的框，取两者的最小值，相当于min函数
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        # pre_nms_limit=tf.mininum(self.config.PRE_NMS_LIMIT,tf.shape(anchors)[1])
        # 获得这些框的索引
        # 这行代码的作用是从scores张量中选择分数最高的pre_nms_limit个元素的索引，并将这些索引
        # 存储在ix变量中。这些索引随后可以用于从原始的锚框集合中选择对应的锚框，以便进行进一步的
        # 处理，如非极大值抑制（NMS）
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        # ix=tf.nn.top_k(scores,pre_nms_limit,sorted=True,name='top_anchors').indices

        # 获得这些框的得分
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        # 获得这些框的调整参数
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        # 获得这些框对应的先验框
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                    self.config.IMAGES_PER_GPU,
                                    names=["pre_nms_anchors"])

        # [batch, N, (y1, x1, y2, x2)]
        # 对先验框进行解码
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])

        # [batch, N, (y1, x1, y2, x2)]
        # 防止超出图片范围
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])


        # 非极大抑制
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # 如果数量达不到设置的建议框数量的话
            # 就padding
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        # def nms(boxes,scores):
        #     indices=tf.image.non_max_suppression(
        #         boxes,scores,self.proposal_count,
        #         self.nms_threshold,name='rpn_non_max_suppression')
        #     proposals=tf.gather(boxes,indices)
        #     # 如果数量达不到设置的建议框数量的话就padding
        #     padding=tf.maximum(self.proposal_count-tf.shape(proposals)[0],0)
        #     proposals=tf.pad(proposals,[(0,padding),(0,0)])
        #     return proposals
        #     pass


        proposals = utils.batch_slice([boxes, scores], nms,
                                      self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)




#----------------------------------------------------------#
#   ROIAlign Layer
#   利用建议框在特征层上截取内容
#----------------------------------------------------------#

def log2_graph(x):
    return tf.log(x) / tf.log(2.0)

def parse_image_meta_graph(meta):
    """
    将meta里面的参数进行分割
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }

class PyramidROIAlign(Layer):
    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    # PyramidROIAlign的例子
    # import tensorflow as tf
    #
    # # 假设我们有一个批次中的一张特征图（为简化起见，这里只展示一张）
    # # 假设特征图的形状为 [1, 100, 100, 3]，即批次大小为1，高度为100，宽度为100，通道数为3
    # feature_maps = tf.random.normal([1, 100, 100, 3])
    #
    # # 假设我们有两个潜在的对象区域需要裁剪
    # # 坐标框格式为 [y1, x1, y2, x2]，并且已经归一化（即除以特征图的高度和宽度）
    # level_boxes = tf.constant([[0.2, 0.3, 0.7, 0.8],  # 第一个对象区域
    #                            [0.4, 0.1, 0.9, 0.6]])  # 第二个对象区域
    #
    # # 因为我们只处理了一个批次中的一张图像，所以box_indices可以简单地是0的重复
    # # 但如果有多个图像，则需要相应地调整这个索引
    # box_indices = tf.constant([0, 0])  # 两个裁剪区域都属于第一张图像
    #
    # # 我们希望将裁剪出的区域缩放到 [20, 20] 的尺寸
    # self.pool_shape = [20, 20]
    #
    # # 使用tf.image.crop_and_resize函数
    # cropped_resized_features = tf.image.crop_and_resize(feature_maps, level_boxes, box_indices, self.pool_shape,
    #                                                     method="bilinear")
    #
    # # cropped_resized_features的形状现在应该是 [2, 20, 20, 3]
    # # 这意味着我们有两个裁剪并缩放后的图像区域，每个区域的形状都是 [20, 20, 3]
    #
    # print(cropped_resized_features.shape)  # 输出: (2, 20, 20, 3)

    # inputs：([rois, image_meta] + feature_maps)
    def call(self, inputs):
        # 建议框的位置
        boxes = inputs[0]

        # image_meta包含了一些必要的图片信息
        image_meta = inputs[1]

        # 取出所有的特征层[batch, height, width, channels]
        feature_maps = inputs[2:]

        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1

        # 获得输入进来的图像的大小
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]

        # 这段代码的目的是根据建议框（ROI，Region of Interest）的尺寸来确定它们应该被分配
        # 到特征金字塔中的哪个层级进行后续处理。在目标检测任务中，特征金字塔是一个常用的结构，
        # 它通过在不同尺度的特征图上检测目标来提高检测的准确性。每个层级通常对应着不同尺寸的目标。

        # 举例说明：
        # 假设你有一个800x600的图像（image_shape = [800, 600]），并且有一个ROI，其尺寸为
        # 300x200（h = 300, w = 200）。
        # 图像面积为 800 * 600 = 480000。
        # ROI尺寸的平方根为 sqrt(300 * 200) = sqrt(60000) ≈ 244.95。
        # 归一化尺寸为 244.95 / (224 / sqrt(480000)) ≈ 2.00（这里做了简化计算，实际中会
        # 使用log2_graph）。对数层级（加上偏移量并限制范围）为 4 + round(log2(2.00)) = 4
        # （因为log2(2.00)接近1，但四舍五入后为1，
        # 加上偏移量4后为5，但受限于最大值5，所以仍为5；然而，在这个简化的例子中，我们假设
        # log2_graph的结果直接给出了接近原始尺寸比值的层级索引，即4）。
        # 因此，这个ROI应该被分配到特征金字塔的第5个层级（注意：层级编号可能因实现而异，这里
        # 假设从2开始编号）。

        # 这段代码的目的是根据建议框（ROI，Region of Interest）的尺寸来确定它们应该被分
        # 配到特征金字塔中的哪个层级进行后续处理。在目标检测任务中，特征金字塔是一个常用的
        # 结构，它通过在不同尺度的特征图上检测目标来提高检测的准确性。每个层级通常对应着不
        # 同尺寸的目标。通过建议框的大小找到这个建议框属于哪个特征层
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        # roi_level = tf.math.log2(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        # batch_size, box_num
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        # 分别在P2-P5中进行截取
        # 这段代码是在处理目标检测或图像分割任务中，特别是在使用特征金字塔
        # （Feature Pyramid Networks, FPN）或类似结构时，用于将感兴趣区域
        # （Regions of Interest, RoIs）或建议框（Region Proposals）映射
        # 到不同尺度的特征层上，并对这些区域进行特征提取（通过池化操作）。具体
        # 来说，这段代码的作用如下：
        #
        # 遍历特征层级：通过for i, level in enumerate(range(2, 6))循环，遍
        # 历从第2层到第5层的特征层（这里假设特征金字塔从第2层开始，并且总共有5层）。
        # 每一层特征图对应着不同尺度的目标检测。
        # 分配RoIs到特征层：对于每个特征层级level，使用tf.where(tf.equal(roi_level, level))
        # 找到所有应该被分配到该层级的RoIs的索引。这里roi_level是一个张量，包含了每个RoI
        # 应该被分配到的特征层级信息。然后，使用tf.gather_nd(boxes, ix)根据这些索引从
        # boxes（包含所有RoIs坐标的张量）中收集出对应层级的RoIs坐标。
        # 记录RoIs索引：将每个层级RoIs的索引（ix）添加到box_to_level列表中，以便
        # 后续可能需要使用这些索引进行反向映射或其他操作。
        # 提取RoIs所属的图片索引：由于RoIs可能来自不同的图片（在批量处理时），因此
        # 需要从RoIs的索引中提取出它们所属的图片索引。这是通过tf.cast(ix[:, 0], tf.int32)
        # 实现的，假设ix的第一维是图片索引。
        # 停止梯度下降：使用tf.stop_gradient确保在反向传播时不会对这些RoIs的坐标和所属
        # 图片索引进行梯度计算。这是因为这些索引是离散的，不需要也不应该进行梯度更新。
        # 特征提取：对于每个层级的RoIs，使用tf.image.crop_and_resize函数从对应的特征图
        # feature_maps[i]中提取出RoI区域的特征。这个函数会根据RoIs的坐标和所属图片索引，
        # 从特征图中裁剪出对应区域，并通过双线性插值（或其他指定方法）调整到统一的尺寸
        # （self.pool_shape）。提取出的特征将被添加到pooled列表中，用于后续的分类和回归任务。
        # 总之，这段代码的目的是将RoIs分配到不同尺度的特征层上，并从这些特征层中提取出RoI区
        # 域的特征，以便进行后续的目标检测或图像分割任务。
        for i, level in enumerate(range(2, 6)):
            # 找到每个特征层对应box
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)
            box_to_level.append(ix)

            # 获得这些box所属的图片
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # 停止梯度下降
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # # 遍历从P2到P5的特征图层级
        # for i, level in enumerate(range(2, 6)):
        #     # 找到属于当前层级的所有ROI的索引
        #     ix = tf.where(tf.equal(roi_level, level))
        #
        #     # 使用这些索引从boxes张量中提取出当前层级对应的ROI坐标
        #     level_boxes = tf.gather_nd(boxes, ix)
        #
        #     # 将这些索引添加到box_to_level列表中，以便后续使用
        #     box_to_level.append(ix)
        #
        #     # 提取出box索引中的图片索引部分（假设ix[:, 0]是图片索引）
        #     box_indices = tf.cast(ix[:, 0], tf.int32)
        #
        #     # 停止level_boxes和box_indices的梯度计算，因为它们不需要在反向传播中更新
        #     level_boxes = tf.stop_gradient(level_boxes)
        #     box_indices = tf.stop_gradient(box_indices)
        #
        #     # 对当前层级的特征图进行ROI池化
        #     # 使用双线性插值方法，将ROI区域裁剪并缩放到self.pool_shape指定的尺寸
        #     pooled_feature = tf.image.crop_and_resize(
        #         feature_maps[i], level_boxes, box_indices, self.pool_shape,
        #         method="bilinear")
        #
        #     # 将当前层级的池化结果添加到pooled列表中
        #     pooled.append(pooled_feature)

        pooled = tf.concat(pooled, axis=0)

        # 将顺序和所属的图片进行堆叠
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # box_to_level[:, 0]表示第几张图
        # box_to_level[:, 1]表示第几张图里的第几个框
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        # 进行排序，将同一张图里的某一些聚集在一起
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]

        # 按顺序获得图片的索引
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # 重新reshape为原来的格式
        # 也就是
        # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled



    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


#----------------------------------------------------------#
#   Detection Layer
#   
#----------------------------------------------------------#

def refine_detections_graph(rois, probs, deltas, window, config):
    # """细化分类建议并过滤重叠部分并返回最终结果探测。
    # Inputs:
    #     rois: [N, (y1, x1, y2, x2)] in normalized coordinates
    #     probs: [N, num_classes]. Class probabilities.
    #     deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
    #             bounding box deltas.
    #     window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
    #         that contains the image excluding the padding.
    #
    # Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
    #     coordinates are normalized.
    # """

    # 找到得分最高的类
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # 序号+类
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    # 取出成绩
    class_scores = tf.gather_nd(probs, indices)
    # 还有框的调整参数
    deltas_specific = tf.gather_nd(deltas, indices)
    # 进行解码
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # 防止超出0-1
    refined_rois = clip_boxes_graph(refined_rois, window)

    # 去除背景
    keep = tf.where(class_ids > 0)[:, 0]
    # 去除背景和得分小的区域
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # 获得除去背景并且得分较高的框还有种类与得分
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):

        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]

        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.DETECTION_MAX_INSTANCES,
                iou_threshold=config.DETECTION_NMS_THRESHOLD)

        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))

        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)

        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. 进行非极大抑制
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. 找到符合要求的需要被保留的建议框
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]

    # 寻找得分最高的num_keep个框
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # 如果达不到数量的话就padding
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections

def norm_boxes_graph(boxes, shape):
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)

class DetectionLayer(Layer):

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # 找到window的小数形式
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 6)


#----------------------------------------------------------#
#   Detection Target Layer
#   该部分代码会输入建议框
#   判断建议框和真实框的重合情况
#   筛选出内部包含物体的建议框
#   利用建议框和真实框编码
#   调整mask的格式使得其和预测格式相同
#----------------------------------------------------------#

def overlaps_graph(boxes1, boxes2):
    """
    用于计算boxes1和boxes2的重合程度
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    返回 [len(boxes1), len(boxes2)]
    """
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # 移除之前获得的padding的部分
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # 计算建议框和所有真实框的重合程度 [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # 计算和 crowd boxes 的重合程度 [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. 正样本建议框和真实框的重合程度大于0.5
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. 负样本建议框和真实框的重合程度小于0.5，Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # 进行正负样本的平衡
    # 取出最大33%的正样本
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # 保持正负样本比例
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # 获得正样本和负样本
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # 获取建议框和真实框重合程度
    positive_overlaps = tf.gather(overlaps, positive_indices)
    
    # 判断是否有真实框
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )
    # 找到每一个建议框对应的真实框和种类
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # 解码获得网络应该有得预测结果
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # 切换mask的形式[N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    
    # 取出对应的层
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # 防止resize后的结果不是1或者0
    masks = tf.round(masks)

    # 一般传入config.TRAIN_ROIS_PER_IMAGE个建议框进行训练，
    # 如果数量不够则padding
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks

def trim_zeros_graph(boxes, name='trim_zeros'):
    """
    如果前一步没有满POST_NMS_ROIS_TRAINING个建议框，会有padding
    要去掉padding
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros

class DetectionTargetLayer(Layer):
    """找到建议框的ground_truth

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)]建议框
    gt_class_ids: [batch, MAX_GT_INSTANCES]每个真实框对应的类
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]真实框的位置
    gt_masks: [batch, height, width, MAX_GT_INSTANCES]真实框的语义分割情况

    Returns: 
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]内部真实存在目标的建议框
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]每个建议框对应的类
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]每个建议框应该有的调整参数
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]每个建议框语义分割情况
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # 对真实框进行编码
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]

