from keras import backend as K
'''
锚点相关配置
'''
class Config:
    def __init__(self):
        self.anchor_box_scales = [128, 256, 512] # 三个框的尺度大小
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]] #三种框的长宽比
        self.rpn_stride = 16 # 表示每隔16个像素放一个锚点
        self.num_rois = 32 # 表示每次处理32个锚点
        self.verbose = True
        self.model_path = "logs/model.h5"
        self.rpn_min_overlap = 0.3 # 表示如果锚点与真实框的IoU小于0.3，则该锚点被视为负样本。
        self.rpn_max_overlap = 0.7 # 如果锚点与真实框的IoU大于0.7，则该锚点被视为正样本。
        self.classifier_min_overlap = 0.1 #如果ROI与真实框的IoU小于这个0.1，则该ROI被视为负样本。

        self.classifier_max_overlap = 0.5 #表示如果ROI与真实框的IoU大于0.5，则该ROI被视为正样本。
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0] #用于标准化分类器回归目标的标准差，regr
        