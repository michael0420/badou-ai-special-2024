
# 定义配置类，用于存储和访问模型训练和检测过程中的各种参数
class Config:
    def __init__(self):
        # 定义用于生成anchor box的尺度
        self.anchor_box_scales = [128, 256, 512]
        # 定义用于生成anchor box的比例
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
        # 定义RPN（Region Proposal Network）的步长
        self.rpn_stride = 16
        # 定义每批生成的ROIs（Region of Interests）的数量
        self.num_rois = 32
        # 定义是否开启详细日志输出
        self.verbose = True
        # 定义模型的保存路径
        self.model_path = "logs/model.h5"
        # 定义RPN中anchor box与ground truth overlap的最小值
        self.rpn_min_overlap = 0.3
        # 定义RPN中anchor box与ground truth overlap的最大值
        self.rpn_max_overlap = 0.7
        # 定义分类器中anchor box与ground truth overlap的最小值
        self.classifier_min_overlap = 0.1
        # 定义分类器中anchor box与ground truth overlap的最大值
        self.classifier_max_overlap = 0.5
        # 定义分类器回归目标的标准差
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
