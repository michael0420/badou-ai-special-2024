import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# 本文件为生成anchors的文件，用于生成anchors
# 本次不调用
# 生成anchors的方法为k-means聚类

"""
转换box的格式，从[x1, y1, x2, y2]转换为[x, y, w, h]
输入：size: 原始图像大小，box: 标注box的信息
输出：x, y, w, h 标注box和原始图像的比值
"""
def convert_coco_bbox(size, box):
    dw = 1. / size[0]                                         # 原始图像的宽
    dh = 1. / size[1]                                         # 原始图像的高
    x = (box[0] + box[2]) / 2.0 - 1                           # box的中心点x
    y = (box[1] + box[3]) / 2.0 - 1                           # box的中心点y
    w = box[2]                                                # box的宽
    h = box[3]                                                # box的高
    x = x * dw                                                # box的中心点x和原始图像的比值
    w = w * dw                                                # box的宽和原始图像的比值
    y = y * dh                                                # box的中心点y和原始图像的比值
    h = h * dh                                                # box的高和原始图像的比值
    return x, y, w, h                                         # 返回box的中心点x和原始图像的比值，box的中心点y和原始图像的比值，box的宽和原始图像的比值，box的高和原始图像的比值

"""
计算每个box和聚类中心的距离值
输入：boxes: 所有的box数据, clusters: 聚类中心
输出：每个box和聚类中心的距离值
"""
def box_iou(boxes, clusters):                                                                               # 计算每个box和聚类中心的距离值
    box_num = boxes.shape[0]                                                                                # box的个数，shape[0]表示行数
    cluster_num = clusters.shape[0]                                                                         # 聚类中心的个数，shape[0]表示行数
    box_area = boxes[:, 0] * boxes[:, 1]                                                                    # 计算每个box的面积，boxes[:, 0]为box的宽，boxes[:, 1]为box的高

    box_area = box_area.repeat(cluster_num)                                                                 # 每个box的面积重复9次，对应9个聚类中心
    box_area = np.reshape(box_area, [box_num, cluster_num])                                                 # 将box_area的形状变为[box_num, cluster_num]

    cluster_area = clusters[:, 0] * clusters[:, 1]                                                          # 计算每个聚类中心的面积
    cluster_area = np.tile(cluster_area, [1, box_num])                                                      # 将cluster_area的形状变为[1, box_num]
    cluster_area = np.reshape(cluster_area, [box_num, cluster_num])                                         # 将cluster_area的形状变为[box_num, cluster_num]

    #这里计算两个矩形的iou，默认所有矩形的左上角坐标都是在原点，然后计算iou，因此只需取长宽最小值相乘就是重叠区域的面积
    boxes_width = np.reshape(boxes[:, 0].repeat(cluster_num), [box_num, cluster_num])                       # 将boxes[:, 0]重复9次，对应9个聚类中心，然后将形状变为[box_num, cluster_num]
    clusters_width = np.reshape(np.tile(clusters[:, 0], [1, box_num]), [box_num, cluster_num])              # 将clusters[:, 0]重复box_num次，对应box_num个聚类中心，然后将形状变为[box_num, cluster_num]
    min_width = np.minimum(clusters_width, boxes_width)                                                     # 计算boxes_width和clusters_width的最小值

    boxes_high = np.reshape(boxes[:, 1].repeat(cluster_num), [box_num, cluster_num])                        # 将boxes[:, 1]重复9次，对应9个聚类中心，然后将形状变为[box_num, cluster_num]
    clusters_high = np.reshape(np.tile(clusters[:, 1], [1, box_num]), [box_num, cluster_num])               # 将clusters[:, 1]重复box_num次，对应box_num个聚类中心，然后将形状变为[box_num, cluster_num]
    min_high = np.minimum(clusters_high, boxes_high)                                                        # 计算boxes_high和clusters_high的最小值

    iou = np.multiply(min_high, min_width) / (box_area + cluster_area - np.multiply(min_high, min_width))   # 计算iou，np.multiply为矩阵对应元素相乘
    return iou                                                                                              # 返回iou

"""
计算所有box和聚类中心的最大iou均值作为准确率
输入：boxes: 所有的box数据, clusters: 聚类中心
输出：所有box和聚类中心的最大iou均值作为准确率
"""
def avg_iou(boxes, clusters):                                                                               # 计算所有box和聚类中心的最大iou均值作为准确率
    return np.mean(np.max(box_iou(boxes, clusters), axis =1))

"""
根据所有box的长宽进行Kmeans聚类
输入：boxes: 所有的box的长宽, cluster_num: 聚类的数量, iteration_cutoff: 当准确率不再降低多少轮停止迭代, function: 聚类中心更新的方式
输出：聚类中心box的大小
"""
def Kmeans(boxes, cluster_num, iteration_cutoff = 25, function = np.median):                       # 根据所有box的长宽进行Kmeans聚类
    boxes_num = boxes.shape[0]                                                                     # box的个数，shape[0]表示行数
    best_average_iou = 0                                                                           # 最好的平均iou
    best_avg_iou_iteration = 0                                                                     # 最好的平均iou的迭代次数
    best_clusters = []                                                                             # 最好的聚类中心
    anchors = []                                                                                   # 聚类中心
    np.random.seed()                                                                               # 随机种子
    # 随机选择所有boxes中的box作为聚类中心
    clusters = boxes[np.random.choice(boxes_num, cluster_num, replace = False)]                    # 随机选择所有boxes中的box作为聚类中心
    count = 0                                                                                      # 迭代次数
    while True:                                                                                    # 当准确率不再降低多少轮停止迭代
        distances = 1. - box_iou(boxes, clusters)                                                  # 计算所有box和聚类中心的iou
        boxes_iou = np.min(distances, axis=1)                                                      # 获取每个box距离哪个聚类中心最近
        # 获取每个box距离哪个聚类中心最近
        current_box_cluster = np.argmin(distances, axis=1)                                         # 获取每个box距离哪个聚类中心最近
        average_iou = np.mean(1. - boxes_iou)                                                      # 计算所有box和聚类中心的最大iou均值作为准确率
        if average_iou > best_average_iou:                                                         # 如果准确率大于最好的准确率
            best_average_iou = average_iou                                                         # 更新最好的准确率
            best_clusters = clusters                                                               # 更新最好的聚类中心
            best_avg_iou_iteration = count                                                         # 更新最好的准确率的迭代次数
        # 通过function的方式更新聚类中心
        for cluster in range(cluster_num):                                                         # 遍历所有聚类中心
            clusters[cluster] = function(boxes[current_box_cluster == cluster], axis=0)            # 通过function的方式更新聚类中心
        if count >= best_avg_iou_iteration + iteration_cutoff:                                     # 当准确率不再降低多少轮停止迭代
            break
        print("Sum of all distances (cost) = {}".format(np.sum(boxes_iou)))                        # 打印所有box和聚类中心的iou
        print("iter: {} Accuracy: {:.2f}%".format(count, avg_iou(boxes, clusters) * 100))          # 打印准确率
        count += 1                                                                                 # 迭代次数加1
    for cluster in best_clusters:                                                                  # 遍历所有聚类中心
        anchors.append([round(cluster[0] * 416), round(cluster[1] * 416)])                         # 将聚类中心的长宽添加到anchors中
    return anchors, best_average_iou                                                               # 返回聚类中心box的大小和最好的平均iou


"""
读取coco数据集的标注信息
输入：datasets: 数据集名字列表
输出：coco数据集的标注信息
"""
def load_cocoDataset(annfile):
    data = []
    coco = COCO(annfile)                                                                                    # 读取coco数据集的标注信息
    cats = coco.loadCats(coco.getCatIds())
    coco.loadImgs()
    base_classes = {cat['id'] : cat['name'] for cat in cats}
    imgId_catIds = [coco.getImgIds(catIds = cat_ids) for cat_ids in base_classes.keys()]
    image_ids = [img_id for img_cat_id in imgId_catIds for img_id in img_cat_id ]
    for image_id in image_ids:
        annIds = coco.getAnnIds(imgIds = image_id)
        anns = coco.loadAnns(annIds)
        img = coco.loadImgs(image_id)[0]
        image_width = img['width']
        image_height = img['height']

        for ann in anns:
            box = ann['bbox']
            bb = convert_coco_bbox((image_width, image_height), box)
            data.append(bb[2:])
    return np.array(data)

"""
    主处理函数
"""
def process(dataFile, cluster_num, iteration_cutoff = 25, function = np.median):
    """
    Introduction
    ------------
        主处理函数
    Parameters
    ----------
        dataFile: 数据集的标注文件
        cluster_num: 聚类中心数目
        iteration_cutoff: 当准确率不再降低多少轮停止迭代
        function: 聚类中心更新的方式
    """
    last_best_iou = 0
    last_anchors = []
    boxes = load_cocoDataset(dataFile)
    box_w = boxes[:1000, 0]
    box_h = boxes[:1000, 1]
    plt.scatter(box_h, box_w, c = 'r')
    anchors = Kmeans(boxes, cluster_num, iteration_cutoff, function)
    plt.scatter(anchors[:,0], anchors[:, 1], c = 'b')
    plt.show()
    for _ in range(100):
        anchors, best_iou = Kmeans(boxes, cluster_num, iteration_cutoff, function)
        if best_iou > last_best_iou:
            last_anchors = anchors
            last_best_iou = best_iou
            print("anchors: {}, avg iou: {}".format(last_anchors, last_best_iou))
    print("final anchors: {}, avg iou: {}".format(last_anchors, last_best_iou))



if __name__ == '__main__':
    process('./annotations/instances_train2014.json', 9)