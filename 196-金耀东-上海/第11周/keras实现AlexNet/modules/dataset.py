"""
训练数据的相关操作
"""
import random , csv
import numpy as np
from keras.utils.np_utils import to_categorical
from modules.global_params import  NUM_CLASS
from keras.preprocessing.image import load_img, img_to_array

def get_imgpaths_labels(datasheet_path):
    '''从datasheet文件里读取图片路径与标签'''
    with open(datasheet_path, 'r') as f:
        reader = csv.reader(f)
        img_paths , labels = [] , []
        for [img_path , label] in reader:
            img_paths.append(img_path) , labels.append(label)
    return img_paths , labels

def generator(img_paths, labels, batch_size, img_size, is_shuffle=True):
    '''数据生成器, 从给定的数据集中生成批量图片与标签'''
    num_expamples , i = len(labels) , 0
    indices = list(range(num_expamples))
    while True:
        img_batch, label_batch = [], []
        if is_shuffle and i == 0:
            random.shuffle(indices)
        for _ in range(batch_size):
            img = img_to_array( load_img(img_paths[indices[i]], target_size=img_size) ) / 255.0
            img_batch.append( img )
            label_batch.append( labels[indices[i]] )
            i = (i+1) % num_expamples
        yield np.array(img_batch) , np.array( to_categorical(label_batch, num_classes=NUM_CLASS) )



