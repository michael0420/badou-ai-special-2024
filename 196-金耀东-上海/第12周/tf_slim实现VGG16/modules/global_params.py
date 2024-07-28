"""
定义全局参数
"""
import os

# 定义各种文件路径
PATH_BASE = os.getcwd() # 获取工作目录
PATH_DATA_DIR = os.path.join(PATH_BASE, "data") # 训练数据与测试路径存放在data下
PATH_LOGS_DIR = os.path.join(PATH_BASE, "logs") # 训练模型存放在logs下
PATH_SYNSET = os.path.join(PATH_DATA_DIR,  "synset.txt") # synset.txt存放路径
PATH_LAST_MODEL = os.path.join(PATH_LOGS_DIR, "vgg_16.ckpt") # 最终模型存放路径

IMG_SHAPE = (224, 224, 3) #输入图片shape:（height, width, channels)