import sys
from operator import itemgetter
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
    # 引申优化项  = resize(h*500/min(h,w), w*500/min(h,w))
    if min(w,h)>500:
        pr_scale = 500.0/min(h,w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)
    elif max(w,h)<500:
        pr_scale = 500.0/max(h,w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)

    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h,w)
    while minl >= 12:
        scales.append(pr_scale*pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales

#-------------------------------------#
#   对pnet处理后的结果进行处理
#-------------------------------------#
def detect_face_12net(cls_prob,roi,out_side,scale,width,height,threshold):

    #调整数组维度，cls_prob 和 roi 的维度被调整，以便后续处理。
    cls_prob = np.swapaxes(cls_prob, 0, 1)
    roi = np.swapaxes(roi, 0, 2)
    #计算步长,stride 是步长，用于将特征图上的坐标映射回原图
    stride = 0
    # stride略等于2
    if out_side != 1:
        stride = float(2*out_side-1)/(out_side-1)
    (x,y) = np.where(cls_prob>=threshold)

    #筛选出置信度高于阈值的区域,x 和 y 是特征图上置信度高于阈值的坐标，boundingbox 是这些坐标的集合。
    boundingbox = np.array([x,y]).T
    # 找到对应原图的位置,bb1 和 bb2 是将特征图上的坐标映射回原图后的边界框坐标。
    bb1 = np.fix((stride * (boundingbox) + 0 ) * scale)
    bb2 = np.fix((stride * (boundingbox) + 11) * scale)
    # plt.scatter(bb1[:,0],bb1[:,1],linewidths=1)
    # plt.scatter(bb2[:,0],bb2[:,1],linewidths=1,c='r')
    # plt.show()
    boundingbox = np.concatenate((bb1,bb2),axis = 1)

    #提取 ROI 偏移量和置信度：dx1, dx2, dx3, dx4 是 ROI 的偏移量，score 是置信度。
    dx1 = roi[0][x,y]
    dx2 = roi[1][x,y]
    dx3 = roi[2][x,y]
    dx4 = roi[3][x,y]
    score = np.array([cls_prob[x,y]]).T
    offset = np.array([dx1,dx2,dx3,dx4]).T

    #调整边界框坐标,boundingbox 是调整后的边界框坐标，rectangles 是包含边界框和置信度的数组，并通过 rect2square 函数将边界框转换为正方形
    boundingbox = boundingbox + offset*12.0*scale
    
    rectangles = np.concatenate((boundingbox,score),axis=1)
    rectangles = rect2square(rectangles)

    #过滤和调整边界框,对每个边界框进行进一步的过滤和调整，确保边界框在图像范围内。
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        sc = rectangles[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,sc])
    return NMS(pick,0.3)
#-----------------------------#
#   将长方形调整为正方形
#-----------------------------#
def rect2square(rectangles):
    #计算矩形的宽度和高度
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    #计算正方形的边长
    l = np.maximum(w,h).T
    #调整矩形的左上角坐标,通过将矩形的中心点移动到正方形的中心点，来调整矩形的左上角坐标
    rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5
    #调整矩形的右下角坐标,通过将左上角坐标加上正方形的边长，来调整矩形的右下角坐标
    rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return rectangles
#-------------------------------------#
#   非极大抑制
#-------------------------------------#
def NMS(rectangles,threshold):
    if len(rectangles)==0:
        return rectangles

    # 将输入的边界框列表转换为 NumPy 数组
    boxes = np.array(rectangles)

    # 提取边界框的各个坐标和置信度
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]

    # 计算每个边界框的面积
    area = np.multiply(x2-x1+1, y2-y1+1)

    # 根据置信度对边界框进行排序
    I = np.array(s.argsort())

    # 用于存储最终保留的边界框索引
    pick = []
    while len(I)>0:
        # 计算当前边界框与其他边界框的交集坐标
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) #I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])

        # 计算交集的宽度和高度
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        # 计算交集面积
        inter = w * h
        # 计算 IoU（交并比）
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])

        # 保留 IoU 小于阈值的边界框
        I = I[np.where(o<=threshold)[0]]

    # 返回保留的边界框
    result_rectangle = boxes[pick].tolist()
    return result_rectangle


#-------------------------------------#
#   对Rnet处理后的结果进行处理
#-------------------------------------#
def filter_face_24net(cls_prob,roi,rectangles,width,height,threshold):
    #提取置信度，cls_prob 是一个包含分类概率的数组，prob 提取了第二列（通常是“人脸”类的概率），pick 是一个索引数组，包含了置信度高于阈值的元素的索引
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)

    #提取边界框坐标，rectangles 是一个包含边界框坐标的数组，通过 pick 索引提取出置信度高于阈值的边界框坐标。
    rectangles = np.array(rectangles)

    x1  = rectangles[pick,0]
    y1  = rectangles[pick,1]
    x2  = rectangles[pick,2]
    y2  = rectangles[pick,3]

    #提取置信度分数
    sc  = np.array([prob[pick]]).T

    #提取 ROI 偏移量，roi 是一个包含 ROI 偏移量的数组，通过 pick 索引提取出相应的偏移量。
    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]

    #计算边界框的宽度和高度：w 和 h 分别是边界框的宽度和高度。
    w   = x2-x1
    h   = y2-y1

    # 调整边界框坐标，根据 ROI 偏移量调整边界框的坐标。
    x1  = np.array([(x1+dx1*w)[0]]).T
    y1  = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2  = np.array([(y2+dx4*h)[0]]).T

    #合并边界框坐标和置信度分数，将调整后的边界框坐标和置信度分数合并成一个新的数组，并通过 rect2square 函数将边界框转换为正方形。
    rectangles = np.concatenate((x1,y1,x2,y2,sc),axis=1)
    rectangles = rect2square(rectangles)
    #过滤和调整边界框，对每个边界框进行进一步的过滤和调整，确保边界框在图像范围内。
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        sc = rectangles[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,sc])

    #使用 NMS 进一步过滤重叠的边界框
    return NMS(pick,0.3)
#-------------------------------------#
#   对onet处理后的结果进行处理
#-------------------------------------#
def filter_face_48net(cls_prob,roi,pts,rectangles,width,height,threshold):
    #提取置信度
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    rectangles = np.array(rectangles)
    #提取边界框坐标，x1, y1, x2, y2 分别是边界框的左上角和右下角坐标
    x1  = rectangles[pick,0]
    y1  = rectangles[pick,1]
    x2  = rectangles[pick,2]
    y2  = rectangles[pick,3]

    #提取置信度分数，sc 是置信度分数
    sc  = np.array([prob[pick]]).T
    #提取 ROI 偏移量
    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]
    #计算宽度和高度，w 和 h 分别是边界框的宽度和高度
    w   = x2-x1
    h   = y2-y1
    #提取关键点坐标，pts0 到 pts9 是关键点坐标
    pts0= np.array([(w*pts[pick,0]+x1)[0]]).T
    pts1= np.array([(h*pts[pick,5]+y1)[0]]).T
    pts2= np.array([(w*pts[pick,1]+x1)[0]]).T
    pts3= np.array([(h*pts[pick,6]+y1)[0]]).T
    pts4= np.array([(w*pts[pick,2]+x1)[0]]).T
    pts5= np.array([(h*pts[pick,7]+y1)[0]]).T
    pts6= np.array([(w*pts[pick,3]+x1)[0]]).T
    pts7= np.array([(h*pts[pick,8]+y1)[0]]).T
    pts8= np.array([(w*pts[pick,4]+x1)[0]]).T
    pts9= np.array([(h*pts[pick,9]+y1)[0]]).T
    #调整边界框坐标，x1, y1, x2, y2 是调整后的边界框坐标
    x1  = np.array([(x1+dx1*w)[0]]).T
    y1  = np.array([(y1+dx2*h)[0]]).T
    x2  = np.array([(x2+dx3*w)[0]]).T
    y2  = np.array([(y2+dx4*h)[0]]).T
    #合并结果，将调整后的边界框坐标、置信度分数和关键点坐标合并成一个数组
    rectangles=np.concatenate((x1,y1,x2,y2,sc,pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9),axis=1)
    #过滤和调整边界框，对每个边界框进行进一步的过滤和调整，确保边界框在图像范围内
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,rectangles[i][4],
                 rectangles[i][5],rectangles[i][6],rectangles[i][7],rectangles[i][8],rectangles[i][9],rectangles[i][10],rectangles[i][11],rectangles[i][12],rectangles[i][13],rectangles[i][14]])
    return NMS(pick,0.3)
