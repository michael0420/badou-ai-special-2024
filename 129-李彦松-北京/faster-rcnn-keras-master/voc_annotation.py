import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

wd = getcwd() # 获取当前路径
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id)) # 打开xml文件，%(year, image_id)字符串格式化，将year和image_id传入
    tree=ET.parse(in_file) # 解析xml文件，把xml文件转换成树形结构
    root = tree.getroot() # 获取根节点
    if root.find('object')==None:
        return
    list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id)) # 写入图片路径，wd是当前路径，year是年份，image_id是图片id
    for obj in root.iter('object'): # 遍历xml文件中的object节点
        difficult = obj.find('difficult').text # 获取difficult节点的值，text是获取节点的值
        cls = obj.find('name').text # 获取name节点的值
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls) # 获取类别的索引
        xmlbox = obj.find('bndbox') # 获取bndbox节点
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)) # 获取xmin,ymin,xmax,ymax节点的值,即边框的坐标
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id)) # 写入边框坐标和类别索引

    list_file.write('\n')

for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split() # 读取txt文件，strip()去掉首尾空格，split()分割字符串。读取指定年份和子集的数据集文件，获取其中的图像ID
    list_file = open('%s_%s.txt'%(year, image_set), 'w') # 创建txt文件并写入
    for image_id in image_ids:
        convert_annotation(year, image_id, list_file) #将xml文件转换成txt文件
    list_file.close()
