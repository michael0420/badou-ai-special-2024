import CONV_RPN_IOU_CreateAnchor as c
from random import shuffle
import matplotlib.pyplot as plt
import cv2

TRAIN_NUM=256


# 为每一个 anchor box 打类别标签
# anchors: create_train_anchors 生成的 anchor_box
# train_num: 每一张图中参加训练的样本数量
# 返回每一个 anchor box 的标签类型 1: 正, 0: 负: -1: 中立
POS_VAL=1   #正样本
NEG_VAL=0    #负样本
NEUTRAL=-1   #不参与计算loss的样本

def get_rpn_cls_lable(img_shape,anchors,ground_truth,pos_thres=0.7,neg_thres=0.3,train_num=TRAIN_NUM):
    cls_labels=[]   #存放每个 anchor_box 的标签值和对应的 gt 坐标
    iou_matrix=[]   ##暂时用来存放每个 anchor_box 与 每个 gt 的 iou, 后面用来判断是正样本还是负样本
                    # anchor_box 为列, ground_truth 为行, 组合成一个二维列表
                    # 交点就是 第 i 个 anchor_box 与 第 j 个 gt 的 iou
                    # 这样做的目的是方便为 anchor_box 分配一个与之 iou 最大的 ground_truth box
    for a in anchors:
        row_iou=[]  #行，一个 anchor_box 与 所有 gt 的 iou
        for gt in ground_truth:
            iou=c.get_iou(a,gt[0])
            if a[0]<0 or a[1]<0 or a[2]>=img_shape[1] or a[3]>=img_shape[0]:
                iou=-1.0
            row_iou.append(iou)
        iou_matrix.append(row_iou)

#与任意的 ground truth IoU ≥ 0.7 是目标, IoU ≤ 0.3 是背景, 0.3 < IoU < 0.7 这部分不用管, 不参加训练
    for r in iou_matrix:
        max_iou=max(r)  #一行中取最大的iou值
        if(max_iou>=pos_thres):
            gt=ground_truth[r.index(max_iou)][0]
            cls_labels.append((POS_VAL,gt))     #大于pos_thres 算正样本
        elif(max_iou<=neg_thres):
            cls_labels.append((NEG_VAL,(0,0,0,0)))
        else:
            cls_labels.append((NEUTRAL,(0,0,0,0)))

#如果其中一个 ground truth 没有任何一个 anchor box 与之 IoU ≥ 0.7, 那与之 IoU 最大的那个 anchor box 也算目标, 也就是正样本
    for g in range(len(ground_truth)):
        max_iou=0
        for a in range(len(anchors)):
            if(iou_matrix[a][g]>max_iou):
                max_iou=iou_matrix[a][g]
        if 0<max_iou<pos_thres:
            for a in range(len(anchors)):
                if iou_matrix[a][g]>=max_iou:
                    cls_labels[a]=(POS_VAL,ground_truth[g][0])

    # 取出所有正样本与负样本的序号, 方便计数与打乱处理
    positives=[i for i,x in enumerate(cls_labels) if POS_VAL==x[0]]
    negatives=[i for i,x in enumerate(cls_labels) if NEG_VAL==x[0]]
    shuffle(positives)
    shuffle(negatives)

 # 如果正样本数量超过 train_num // 2, 随机选 train_num // 2 个,
    # 上面打乱后直接取前 train_num // 2 个
    pos_num=min(train_num//2,len(positives))
    for p in positives[pos_num:]:
        cls_labels[p]=(NEUTRAL,(0,0,0,0))
 # 参加训练的负样本的数量
    train_negs=train_num-pos_num
    for n in negatives[train_negs:]:
        cls_labels[n]=(NEUTRAL,(0,0,0,0))  #去除多余负样本

    cls_ids=[]   # 每个 anchor 标签, POS_VAL, NEG_VAL 或 NEUTRAL
    gt_box=[]    # 每个 anchor 对应的 gt 坐标
    for label in cls_labels:
        cls_ids.append(label[0])
        gt_box.append(label[1])
    return cls_ids,gt_box


# 测试 get_rpn_cls_label, 将其画到图像上, 这里 train_num 设置为 32, 方便显示
rpn_cls_label,gt_boxes=get_rpn_cls_lable(c.image.shape,c.anchors,c.gts,train_num=32)
print('positive boxes:',rpn_cls_label.count(POS_VAL))
print('negative boxes:',rpn_cls_label.count(NEG_VAL))

img_copy=c.image.copy()

for i,a in enumerate(c.anchors):
    if POS_VAL==rpn_cls_label[i]:
        gt=gt_boxes[i]
        # 测试 get_rpn_cls_label 带出来的 gt 是否正确
        cv2.rectangle(img_copy,(gt[0],gt[1]),(gt[2],gt[3]),(255,55,55),2)
        cv2.rectangle(img_copy,(a[0],a[1]),(a[2],a[3]),(0,255,0),2)
    elif NEG_VAL==rpn_cls_label[i]:
        cv2.rectangle(img_copy,(a[0],a[1]),(a[2],a[3]),(0,0,255),1)
plt.figure('anchor_box',figsize=(8,4))
plt.imshow(img_copy[...,::-1])
plt.show()



