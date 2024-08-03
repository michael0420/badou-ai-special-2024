from  keras.layers import Conv2D,Input,TimeDistributed,Flatten,Dense,Reshape,BatchNormalization,Activation,MaxPooling2D
from keras.models import Model
import keras
import random
import cv2
import matplotlib.pyplot as plt
import Get_dataset as g


#参数配置
ANCHOR_SIZE=(64,128,256)  #anchor box 的三种边长
ANCHOR_RATIO=(0.5,1.0,2.0) #anchor box 的三种边长比例
ANCHOR_NUM=len(ANCHOR_RATIO)*len(ANCHOR_SIZE)   #每个像素点对应的anchor box 数量
SHORT_SIZE=300   # 图像缩放最短边度长(论文是600, 300 是为了训练速度快一点)
FEATURE_STRIDE=16   ##特征图相对于原始输入图像的缩小的倍数, 如果用 VGG16 作为特征提取网络就是 16
DATA_PATH='F:/data/VOC2007/dataset'   #数据路径
CATEGORIES = ("back_ground",
              "aeroplane", "bicycle", "bird", "boat", "bottle",
              "bus", "car", "cat", "chair", "cow",
              "diningtable", "dog", "horse", "motorbike", "person",
              "pottedplant", "sheep", "sofa", "train", "tvmonitor")  #类别列表



#卷积和池化合并层
def conv_pool(input=None,filters=1,ks=(3,3),dilation_rate=(1,1),padding='same',
              activation='relu',conv_layers=1,pool_enable=True,normalize=False,init=None,name=None):
    if None==init:
        init=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=random.randint(0,1024))

    x=input
    for i in range(max(conv_layers,1)):
        layer_name=None if None==name else name+'_'+str(i+1)

        if normalize:
            x=Conv2D(filters=filters,kernel_size=ks,dilation_rate=dilation_rate,  #kernel_initializer 卷积核初始化
                     kernel_initializer=init,padding=padding,name=layer_name)  #dilation：卷积核中元素之间的间距，用于控制感受野的大小。默认值为1。
            x=BatchNormalization()(x)
            x=Activation('relu')(x)
        else:
            x=Conv2D(filters=filters,kernel_size=ks,dilation_rate=dilation_rate,
                     kernel_initializer=init,padding=padding,activation=activation,name=layer_name)(x)
    y=MaxPooling2D(pool_size=(2,2),strides=(2,2))(x) if pool_enable else x
    return y

#定义vgg模型结构的conv部分
def vgg16_conv(input_layer):
    x1=conv_pool(input_layer,64,conv_layers=2,name='vgg16_x1')
    x2=conv_pool(x1,128,conv_layers=2,name='vgg16_x2')
    x3=conv_pool(x2,256,conv_layers=3,name='vgg16_x3')
    x4=conv_pool(x3,512,conv_layers=3,name='vgg16_x4')
    x5=conv_pool(x4,512,conv_layers=3,pool_enable=False,name='vgg16_x5')

    return x5   #feature map


#RPN网络
def rpn(feature,anchors=ANCHOR_NUM):
    x=Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu',name='rpn_conv')(feature)  #feature map 通过3*3*512卷积

    rpn_cls=Conv2D(anchors*1,kernel_size=(1,1),activation='sigmoid',kernel_initializer='uniform',name='rpn_cls')(x)

    rpn_reg=Conv2D(anchors*4,kernel_size=(1,1),activation='linear',kernel_initializer='zero',name='rpn_reg')(x)

    return rpn_cls


x=Input(shape=(None,None,3),name='input')
feature=vgg16_conv(x)
rpn_cls,rpn_reg=rpn(feature)
rpn_model=Model(x,[rpn_cls,rpn_reg],name='rpn_model')


rpn_model.summary()




#IOU   anchor box 坐标格式为（x1,y1,,x2,y2） 左上、右下两个坐标点
#交集
def intersection(a,b):
     x=max(a[0],b[0])
     y=max(a[1],b[1])
     w=min(a[2],b[2])-x
     h=min(a[3],b[3])-y

     if w<0 or h<0:
         return 0

     return w*h

#并集
def union(a,b):
    area1=(a[2]-a[0])*(a[3]-a[1])
    area2=(b[2]-b[0])*(b[3]-b[1])
    area_union=area1+area2-intersection(a,b)  #分别计算两个框的面积 再减去交集  即为并集的面积

    return area_union

def get_iou(a,b):
    if a[2]<a[0]:
        a[2],a[0]=a[0],a[2]
    if a[3]<a[1]:
        a[3],a[1]=a[1],a[3]
    if b[2]<b[0]:
        b[2],b[0]=b[0],b[2]
    if b[3]<b[1]:
        b[3],b[1]=b[1],b[3]   #防止left<right   top<bottom

    area_i=float(intersection(a,b))
    area_u=float(union(a,b))

    if area_u<0 or area_i<0:
        return 0

    return area_i/area_u  #IOU==交集/并集
'''
#测试 IoU
a = (8, 8, 32, 64)
b = (3, 3, 32, 65)

print("iou(a, b) =", get_iou(a, b))
'''

#生成anchor box
def create_base_anchors(size=ANCHOR_SIZE,ratios=ANCHOR_RATIO):
    anchors=[]
    # round()方法返回浮点数x的四舍五入值
    for r in ratios:
        #各种比例下的边长
       side1=[round((x*x*r)**0.5) for x in size]
       side2=[round(s/r) for s in side1]
       for i in range(len(size)):
          anchors.append((-side1[i]//2,-side2[i]//2,side1[i]//2,side2[i]//2))
    return anchors

# 测试基础 anchor box
base_anchors = create_base_anchors()
for a in base_anchors:
    print(a, "    w =", a[2] - a[0], "h =", a[3] - a[1])

#图像缩放函数  把feature map 中的点对应到原图上
def new_size_image(image,short_size=SHORT_SIZE):  ##short_size  图像缩放最短边长，此处设置不能小于300
    img_shape=list(image.shape)
    scale=1.0
    if img_shape[0]<img_shape[1]:
        scale=short_size/img_shape[0]
        img_shape[0]=short_size
        img_shape[1]=round(img_shape[1]*scale)
    else:
        scale=short_size/img_shape[1]
        img_shape[1]=short_size
        img_shape[0]=round(img_shape[0]*scale)

    new_image=cv2.resize(image,(img_shape[1],img_shape[0]),interpolation=cv2.INTER_LINEAR)  #原图的长宽按比例缩放

    return new_image,scale  #返回缩放后的图片及比例

#在原图生成anchor box
# feature_size: 特征图尺寸
# anchors: k 个基础 anchor box 坐标
# stride: 图像到特征图缩小倍数
def create_train_anchor(feature_size,base_anchors,stride=FEATURE_STRIDE):
    anchors=[]
    for r in range(feature_size[0]):
        for c in range(feature_size[1]):
            for a in base_anchors:
                anchors.append([
                    c*stride+stride//2+a[0],
                    r*stride+stride//2+a[1],
                    c*stride+stride//2+a[2],
                    r*stride+stride//2+a[3]
                ])
    return anchors


train_set,valid_set,test_set=g.get_data_set(DATA_PATH,(0.8,0.1,0.1))

print("Total number:", len(train_set) + len(valid_set) + len(test_set),
      " Train number:", len(train_set),
      " Valid number:", len(valid_set),
      " Test number:", len(test_set))

print("First element:", train_set[0])

#测试画框效果
idx=random.randint(0,len(train_set))
img= cv2.imread(train_set[idx][0]) #读图
image,scale=new_size_image(img,SHORT_SIZE)  #缩放尺寸
feature_size=(image.shape[0]//FEATURE_STRIDE,image.shape[1]//FEATURE_STRIDE)
print('image_size:',image.shape,'feature_size:',feature_size)
anchors=create_train_anchor(feature_size,base_anchors,FEATURE_STRIDE)
print('anchor num:',len(anchors))   #输出anchor总数
#选一个点画anchor
center=((feature_size[0]//2)*feature_size[1]+feature_size[1]//2)*len(base_anchors)
print('center',center)
colors=((0,0,255),(0,255,0),(255,0,0))
img_copy=image.copy()
for i,a in enumerate(anchors[center:center+len(base_anchors)]):
    cv2.rectangle(img_copy,(a[0],a[1]),(a[2],a[3]),colors[i%3],2)

plt.figure('anchor box',figsize=(8,4))
plt.imshow(img_copy[...,::-1])
plt.show()




image = cv2.imread(train_set[idx][0])
label_data=train_set[idx]
print(label_data)
gts=g.get_ground_truth(label_data[1],label_data[2],CATEGORIES)
print(gts)
img_copy=image.copy()
for gt in gts:
    gt[0][0]=round(gt[0][0]*scale)
    gt[0][1]=round(gt[0][1]*scale)
    gt[0][2]=round(gt[0][2]*scale)
    gt[0][3]=round(gt[0][3]*scale)
    print(gt,'class:',CATEGORIES[gt[1]])

    cv2.rectangle(img_copy,(gt[0][0],gt[0][1]),(gt[0][2],gt[0][3]),(0,250,0),5)

plt.figure("label_box", figsize = (8, 4))
plt.imshow(img_copy[..., : : -1]) # 这里的通道要反过来显示才正常
plt.show()

























