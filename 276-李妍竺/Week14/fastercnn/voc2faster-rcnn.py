#-------------------------------------------------------------#
#   这份代码用于将 VOC2007 数据集的标注文件进行分割
#-------------------------------------------------------------#

import os    #它提供与操作系统交互的功能，如文件和目录操作
import random 
 
xmlfilepath=r'./VOCdevkit/VOC2007/Annotations' #设置 XML 文件的路径，这些 XML 文件包含图像的标注信息。
saveBasePath=r"./VOCdevkit/VOC2007/ImageSets/Main/" #设置保存分割文件的路径。
# 设置数据分割比例
trainval_percent=1 #将数据集的 100% 用于训练和验证（trainval）
train_percent=1 #trainval 数据中用于训练的比例为 100%。

temp_xml = os.listdir(xmlfilepath)  #annotation
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

# 计算数据集大小
num=len(total_xml)  
list=range(num)   ## 创建一个范围为 0 到 num-1 的迭代器，表示 XML 文件的索引。

tv=int(num*trainval_percent)  #计算用于验证的数据数量（trainval）。
tr=int(tv*train_percent)  #计算用于训练的数据数量。
trainval= random.sample(list,tv)  #从 list 中随机选择 tv 个元素，组成 trainval 列表，用于验证。
train=random.sample(trainval,tr)  #从 trainval 列表中随机选择 tr 个元素，组成 train 列表，专门用于训练。
 
print("train and val size",tv)
print("traub suze",tr)

# 保存数据集分割文件
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
# 保存数据集文件名：
for i  in list:  
    name=total_xml[i][:-4]+'\n'  
    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    else:  
        ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()
