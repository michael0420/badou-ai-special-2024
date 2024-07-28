"""
定义全局参数
"""
import os

# 路径
PATH_BASE = os.getcwd() # 获取工作目录
PATH_DATA_DIR = os.path.join(PATH_BASE, "data") # 训练数据与测试路径存放在data下
PATH_LOGS_DIR = os.path.join(PATH_BASE, "logs") # 训练模型存放在logs下
PATH_TRIAN_IMGS_DIR = os.path.join(PATH_DATA_DIR, "train", "imgs") # 训练图片存放路径
PATH_TEST_IMGS_DIR = os.path.join(PATH_DATA_DIR, "test", "imgs") # 测试图片存放路径
PATH_TRIAN_DATASHEET = os.path.join(PATH_DATA_DIR, "train", "datasheet.csv") # 训练数据的我datasheet文件存放路径
PATH_TEST_DATASHEET = os.path.join(PATH_DATA_DIR, "test", "datasheet.csv") # 测试数据的datasheet文件存放路径
PATH_LAST_MODEL = os.path.join(PATH_LOGS_DIR, "last.h5") # 最终模型存放路径

IMG_SHAPE = (224, 224, 3) #输入图片shape:（height, width, channels)
NUM_CLASS = 2
CLASSES = ['cat', 'dog']
