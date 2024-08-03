import xml.etree.ElementTree as ET # 用于解析 XML 文件。
from os import getcwd  # 用于获取当前工作目录（Working Directory）的路径

# 定义数据集和类别
sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')] #定义了三个数据集，分别对应 VOC2007 的训练集、验证集和测试集

wd = getcwd() # 获取当前工作目录的路径，并储存在 wd 变量中
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def convert_annotation(year, image_id, list_file):  #定义一个函数将 XML 标注转换为训练所用的格式
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id)) #根据年份和图像 ID 打开相应的 XML 文件。
    tree=ET.parse(in_file)  #使用 ElementTree 解析 XML 文件。
    root = tree.getroot()  #获取 XML 文件的根节点。

    # 处理对象标注：
    if root.find('object')==None:
        return
    list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))

    # 迭代处理每个对象并输出其标注信息：
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id)) #获取边界框的四个坐标。将边界框和类别信息写入到输出文件

    list_file.write('\n')

# 迭代处理数据集并调用转换函数：
for year, image_set in sets:  # 迭代处理每个数据集（训练集、验证集、测试集）。
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split() #打开对应的数据集文本文件，获取其中列出的图像 ID
    list_file = open('%s_%s.txt'%(year, image_set), 'w') #打开或创建一个文件，用于保存处理后的图像和标注信息。
    for image_id in image_ids:
        convert_annotation(year, image_id, list_file)  #迭代处理每个图像 ID，调用 convert_annotation 函数进行转换。
    list_file.close()
