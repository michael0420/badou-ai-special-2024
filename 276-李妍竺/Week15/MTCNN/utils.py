import sys  #与python解释器交互的接口
from operator import itemgetter  #获取对象指定域中的值
import numpy as np
import cv2
import matplotlib.pyplot as plt

#-----------------------------#
#   计算原始输入图像
#   每一次缩放的比例
#-----------------------------#

def calculateScales(img):
    copy_img = img.copy()
    pr_scale = 1.0
    h,w,_ = copy_img.shape
    #引申优化项  resize(h*500/min(h,w),w*500/min(h,w))
    if min(w,h) >500:
        pr_scale = 500.0/min(h,w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)
    elif max(w,h)<500:
        pr_scale = 500.0 / max(h, w)
        w = int(w * pr_scale)
        h = int(h * pr_scale)

    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h,w)
    while minl >=12:
        scales.append(pr_scale*pow(factor,factor_count)) #比例缩小
        minl *=factor
        factor_count += 1
    return scales

#-------------------------------------#
#   对pnet处理后的结果进行处理
#-------------------------------------#

def detect_face_12net(cls_prob,roi,out_side,scale,width,height,threshold):
    # hw -> wh
    cls_prob = np.swapaxes(cls_prob,0,1) #swapaxes(a,axis1,axis2) 交换数组中的两个轴
    # hwc-cwh
    roi = np.swapaxes(roi,0,2)

    stride = 0

    if stride != 1:
        stride = float(2*out_side -1)/(out_side-1)
    (x,y) = np.where(cls_prob >= threshold)  #where(condition):输出满足条件元素的坐标
    # 保证每一行是一个x，y
    bounding_box = np.array([x, y]).T
    # 找到对应原图的位置，（左上和右下）
    bb1 = np.fix((stride * bounding_box + 0 )*scale) #向0取整
    bb2 = np.fix((stride * bounding_box + 11) * scale)

    #plt.scatter(bb1[:,0],bb1[:,1],linewidths=1)
    #plt.scatter(bb2[:,0],bb2[:,1],linewidths=1,c='r')
    #plt.show()

    # 水平方向拼接，每一行表示一个框，这四个值就表示这个框两个点的坐标
    bounding_box = np.concatenate((bb1, bb2), axis=1)

    dx1 = roi[0][x,y]
    dx2 = roi[1][x, y]
    dx3 = roi[2][x, y]
    dx4 = roi[3][x, y]

    score = np.array([cls_prob[x, y]]).T
    offset = np.array([dx1, dx2, dx3, dx4]).T
    # 修正后的人脸检测框
    bounding_box = bounding_box + offset*12.0*scale
    # rectangles: [x1,y1,x2,y2,p]
    rectangles = np.concatenate((bounding_box, score), axis=1)
    rectangles = rect2square(rectangles) #将长方形转成正方形

    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0,rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2>y1:
            pick.append([x1,y1,x2,y2,sc])
    return nms(pick,0.3)

# -----------------------------#
#   将长方形调整为正方形
# -----------------------------#
def rect2square(rectangles):
    w = rectangles[:,2] - rectangles[:,0] #x2-x1
    h = rectangles[:,3] - rectangles[:,1] #y2-y1
    l = np.maximum(w,h).T

    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T
    return rectangles
# -------------------------------------#
#   非极大抑制
# -------------------------------------#

def nms(rectangles,threshold):
    if len(rectangles)==0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]
    # 每个边界框的面积，+1是确保结果不会为0
    area = np.multiply(x2-x1+1, y2-y1+1)
    ind = np.array(s.argsort())
    pick = []
    while len(ind) > 0:
    # ind为按prob顺序排列的下标，最后的一个下标表示prob最大的值的下标
    # 例如xx1，就是将最大概率的边框的左上角的x值与其他概率的边框的左上角的x值相比，取最大
    # 为了后面将最大概率的边框与其他概率的边框做iou
        xx1 = np.maximum(x1[ind[-1]], x1[ind[0:-1]])
        yy1 = np.maximum(y1[ind[-1]], y1[ind[0:-1]])
        xx2 = np.minimum(x2[ind[-1]], x2[ind[0:-1]])
        yy2 = np.minimum(y2[ind[-1]], y2[ind[0:-1]])
        # 最大概率的边框与其他概率的边框的交集
        w = np.maximum(.0, xx2 - xx1 + 1)
        h = np.maximum(.0, yy2 - yy1 + 1)
        inter = w * h
        # 最大概率的边框与其他概率的边框的iou
        o = inter / (area[ind[-1]] + area[ind[0:-1]] - inter)
        pick.append(ind[-1])
        # 更新ind的值，留下与这个概率最大的边框的iou>iou阈值的边框
        # （相当于这个边框尺度的边框只取概率最大的一个）
        ind = ind[np.where(o < threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle

# -------------------------------------#
#   对Rnet处理后的结果进行处理
# -------------------------------------#
def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    # fc输出结果（batch，2）正确的概率
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)
    # 筛选出来的边框的信息
    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]
    sc = np.array([prob[pick]]).T
    # 修正偏移量
    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    w = x2 - x1
    h = y2 - y1
    # 微调后的坐标
    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])
    return nms(pick, 0.3)

# -------------------------------------#
#   对onet处理后的结果进行处理
# -------------------------------------#
def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    # 正确的概率
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    # 筛选出来的边框的信息
    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    sc = np.array([prob[pick]]).T

    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    w = x2 - x1
    h = y2 - y1

    # pts0
    # 存储了关键点0（通常是左眼的位置）相对于(x1, y1)的水平偏移量。
    # pts1
    # 存储了关键点1（通常是左眼的位置）相对于(x1, y1)的垂直偏移量。
    pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
    pts1 = np.array([(h * pts[pick, 5] + y1)[0]]).T
    pts2 = np.array([(w * pts[pick, 1] + x1)[0]]).T
    pts3 = np.array([(h * pts[pick, 6] + y1)[0]]).T
    pts4 = np.array([(w * pts[pick, 2] + x1)[0]]).T
    pts5 = np.array([(h * pts[pick, 7] + y1)[0]]).T
    pts6 = np.array([(w * pts[pick, 3] + x1)[0]]).T
    pts7 = np.array([(h * pts[pick, 8] + y1)[0]]).T
    pts8 = np.array([(w * pts[pick, 4] + x1)[0]]).T
    pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T

    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4,
                                 pts5, pts6, pts7, pts8, pts9), axis=1)

    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, rectangles[i][4],
                         rectangles[i][5], rectangles[i][6], rectangles[i][7], rectangles[i][8], rectangles[i][9],
                         rectangles[i][10], rectangles[i][11], rectangles[i][12], rectangles[i][13], rectangles[i][14]])
    return nms(pick, 0.3)


