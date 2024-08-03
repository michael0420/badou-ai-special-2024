import json
import xml.etree.ElementTree as et
import os
from random import shuffle

# 取得图像和标注文件路径
# data_set_path: 数据集所在路径
# split_rate: 这些文件中用于训练, 验证, 测试所占的比例
#             如果为 None, 则不区分, 直接返回全部
#             如果只写一个小数, 如 0.8, 则表示 80% 为训练集, 20% 为验证集, 没有测试集
#             如果是一个 tuple 或 list, 只有一个元素的话, 同上面的一个小数的情况
# shuffle_enable: 是否要打乱顺序
# 返回训练集, 验证集和验证集路径列表
def get_data_set(data_set_path,spilt_rate=(0.7,0.2,0.1),shuffle_enable=True):
    data_set=[]
    files=os.listdir(data_set_path)
    for f in files:
        ext=os.path.splitext(f)[1]  #os.path.splitext是Python标准库中的一个函数，它可以将一个文件路径拆分成两部分：文件名和文件扩展名
        if ext in('.jpg','.png','.bmp'):
            img_path=os.path.join(data_set_path,f) # os.path.join()能够智能地处理相对路径和绝对路径的拼接
            ann_type='' #标注文件类型
            ann_path=img_path.replace(ext,'.json')

            if os.path.exists(ann_path):  #路径是否存在，返回false或者true
                ann_type='json'
            else:
                ann_path=img_path.replace(ext,'.xml')
                if os.path.exists(ann_path):
                    ann_type='xml'
            if ''==ann_type:
                 continue
            data_set.append((img_path,ann_path,ann_type))

    if shuffle_enable:
        shuffle(data_set)

    if None==spilt_rate:
        return data_set

    total_num=len(data_set)
 #isinstance 是Python中的一个内建函数。是用来判断一个对象的变量类型。
    if isinstance(spilt_rate,float) or 1==len(spilt_rate):
        if isinstance(spilt_rate,float):
            spilt_rate=[spilt_rate]  #转换为列表
        train_pos=int(total_num*spilt_rate[0])
        train_set=data_set[:train_pos]
        valid_set=data_set[train_pos:]

        return train_set,valid_set

    elif isinstance(spilt_rate,tuple) or isinstance(spilt_rate,list):
        list_len=len(spilt_rate)
        assert (list_len)>1    #断言
        train_pos=int(total_num*spilt_rate[0])
        valid_pos=int(total_num*(spilt_rate[0]+spilt_rate[1]))
        train_set=data_set[:train_pos]  #训练集
        valid_set=data_set[train_pos:valid_pos]   #验证集
        test_set=data_set[valid_pos:]  #测试集

        return train_set,valid_set,test_set


# 从 xml 或 json 文件中读出 ground_truth
# data_set: get_data_set 函数返回的列表
# categories: 类别列表
# file_type: 标注文件类型
# 返回 ground_truth 坐标与类别
def  get_ground_truth(label_path,file_tpye,categories):
    ground_truth=[]
    with open(label_path,'r',encoding='utf-8') as f:
        if 'json'==file_tpye:
            jsn=f.read()
            js_dict=json.load(jsn)
            shapes=js_dict["shapes"]

            for shape in shapes:
                if shape["label"] in categories:
                    pts=shape["points"]
                    x1=round(pts[0][0])
                    x2=round(pts[1][0])
                    y1=round(pts[0][1])
                    y2=round(pts[1][1])
                    if x1>x2:
                        x1,x2=x2,x1
                    if y1>y2:
                        y1,y2=y2,y1
                    bnd_box=[x1,y1,x2,y2]
                    cls_id=categories.index(shape["label"])
                    ground_truth.append([bnd_box,cls_id])

        elif 'xml'==file_tpye:
            tree=et.parse(f)   #et.parse  解析xml文件
            root=tree.getroot()
            for obj in root.iter('object'):   #.iter 遍历
                cls_id=obj.find("name").text
                cls_id=categories.index(cls_id)

                bnd_box=obj.find("bndbox")
                bnd_box=[
                    int(bnd_box.find('xmin').text),
                    int(bnd_box.find('ymin').text),
                    int(bnd_box.find('xmax').text),
                    int(bnd_box.find('ymax').text),
                ]
                ground_truth.append([bnd_box,cls_id])

    return ground_truth





