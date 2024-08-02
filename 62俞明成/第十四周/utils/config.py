
class Config:

    def __init__(self):
        """
            anchor_box_scales是一个列表，包含了三个元素，分别表示三个不同大小的anchor box的尺寸。
            anchor_box_ratios是一个列表，包含了三个元素，每个元素都是一个包含两个元素的列表，表示不同长宽比的anchor box的长宽比。
            rpn_stride表示RPN网络的步长。
            num_rois表示每个mini-batch中ROIs的数量。
            verbose表示是否打印详细信息。
            model_path表示模型的路径。
            rpn_min_overlap和rpn_max_overlap分别表示RPN网络中anchor box与ground truth的重叠面积的最小和最大阈值。
            classifier_min_overlap和classifier_max_overlap分别表示分类器中ROIs与ground truth的重叠面积的最小和最大阈值。
            classifier_regr_std是一个列表，包含了四个元素，表示分类器回归四个参数的标准差
        """
        self.anchor_box_scales = [128, 256, 512]
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
        self.rpn_stride = 16
        self.num_rois = 32
        self.verbose = True
        self.model_path = "logs/model.h5"
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
        