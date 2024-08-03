# faster rcnn
## 原理
### 1 Conv layers
**共13conv层，13relu层，4个pooling层**

1. 所有conv层都是：kernel_size=3, pad=1, stride=1  **(不改变尺寸)**
2. 所有pooling层都是：kernel_size=2,pad=1,stride=2 **（尺寸减半）**

**一个MxN大小的矩阵经过Conv layers 固定变为(M/16)x(N/16).(后期方便还原)**

### 2 区域生成网络Region Proposal Networks(RPN)

1. 上面一条通过softmax**分类**anchors，获得**positive和negative**
2. 下面一条用于计算对于有anchors的**bounding box regression偏移量**，以获得精确的proposal
3. 最后的proposal**s综合positive anchors 和对应的偏移量**获得proposals,同时剔除太小或超出边界的proposals.
4. **conv的num_output=18**，也就是经过该卷积的输出图像为：**WxHx18(9* 2 )-**>**方便softmax**
5. softmax前后的reshape 也是为了方便nsoftmax分类。 
6. 生成矩阵为： **[1,18,H,W]** 为了softmax,将其改为：**[1,2,9xH,W]  腾空出一个维度**方便判断 后面再复原
7. 下面一条，**bbox:输出通道36：WxHx36**：每个点都有9个anchors，每个anchors有4个用于回归的变换量： $[d_x(A),d_{y}(A),d_{w}(A),d_{h}(A)]$ 
**8. Proposal layer 4个输入：**
    - positive vs negative 分类器结果
    - 对应的b box reg的变换量
    - im_info=[M,N,scale_factor]：保存此次缩放的所有信息
    - 参数featrue_stride=16 ：之前的/16 用于计算anchor偏移量
**9. Proposal layer 按以下顺序处理：**
    - 利用变换量对所有positive anchors做bbox regression回归
    - 按照输入的positive softmax scores由大到小排序anchors,提取前(eg.6000 按数量的前n个)个anchors,即提取修正后位置后的possitive anchors
    - 对剩余positive anchors进行NMS（非极大值抑制）：按> 多少%
    - 输出proposal

**总结RPN:**

**生成anchors -> softmax分类器提取positive anchors -> bbox reg回归positive anchors ->Proposal layer 生成proposals**

#### anchors

RPN卷积后，对每个像素点，上采样映射到原始图像一个区域，找到这个区域的中心位置，然后选取9种anchor box
9个矩阵共3种面积：128，256，512
3种形状：长宽比：1：1，1：2，2：1

每行四个值表示矩形左上和右下的坐标

**每个点都配备这9种anchors作为初始的检测框**


### 3 ROI pooling
**ROI是为了规范框的大小**
收集proposals，并计算proposal feature maps，送入后序网络
**2个输入：**
1. 原始的feature maps
2. RPN输出的proposal boxes(大小各不相同)

**ROI原理：**
新参数**pooled_w pooled_h spatial_scale(1/16)**
**过程：**
1. 由于proposal是对应MxN尺度的，所以首先使用spatial_scale参数将其映射回(M/16)* (N/16)的尺度
2. 再将每个proposal对应的feature map 区域水平分为pooled_w* pooled_h的网格
3. 对网格的每一份都进行max pooling处理。
处理后，即使大小不同的proposal**输出结果都是pooled_w * pooled_h 固定大小。**

**（不管什么大小的proposal,都均分为7 * 7 = 49    这49个区域都做max pooling 得到49个数。这样大小就都是7 * 7 一样大了。**
### 4 Classification
1. 通过全连接和softmax 对proposals进行分类（识别）
2. 再次对proposals进行bounding box regression，获取更高精度的预测框

## 代码实现
### 训练步骤
1. 数据集的准备
训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。
训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。

2. 数据集的处理
在完成数据集的摆放之后，利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。
修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。
训练自己的数据集时，可以自己建立一个cls_classes.txt，里面写自己所需要区分的类别。
model_data/cls_classes.txt

3. 开始网络训练
classes_path用于指向检测类别所对应的txt，这个txt和voc_annotation.py里面的txt一样。
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。

4. 训练结果预测
训练结果预测需要用到两个文件，分别是frcnn.py和predict.py。在frcnn.py里面修改model_path以及classes_path。
model_path指向训练好的权值文件，在logs文件夹里。
classes_path指向检测类别所对应的txt。
完成修改后运行predict.py进行检测。运行后输入图片路径即可检测。

###预测步骤
在predict.py里面进行设置可以进行fps测试和video视频检测。



