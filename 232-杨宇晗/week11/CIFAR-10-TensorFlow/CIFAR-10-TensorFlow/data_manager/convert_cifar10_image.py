import urllib
import os
import sys
import tarfile
import glob
import pickle
import numpy as np
import cv2


"""
Downloads the `tarball_url` and uncompresses it locally.
Args:
	tarball_url: The URL of a tarball file.
	dataset_dir: The directory where the temporary files are stored.
"""
def download_and_uncompress_tarball(tarball_url, dataset_dir):

  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)

#cifar10的10个分类
classification = ['airplane',
                  'automobile',
                  'bird',
                  'cat',
                  'deer',
                  'dog',
                  'frog',
                  'horse',
                  'ship',
                  'truck']


"""
cifar-10下载后为二进制文件
解析脚本
"""
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


#数据下载URL
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
#存储的文件夹
DATA_DIR = 'data'

#调用下载函数
#download_and_uncompress_tarball(DATA_URL, DATA_DIR)

#指定文件夹的路径
folders = '/root/FaceAI/data_manager/data/cifar-10-batches-py'

#获取当前文件夹下所有的训练样本
trfiles = glob.glob(folders + "/data_batch*")

data  = []
labels = []

for file in trfiles:
    dt = unpickle(file)
	#解析数据并转换为list
    data += list(dt[b"data"])
	#解析labels并转换为list
    labels += list(dt[b"labels"])	

#打印labels，对应classification中数据下标（从0开始）
print(labels)	

# -1 表达当前有多少张图片，具体数值 = data原始维度 / 一张图片的大小
#cifar10图像大小为3通道32*32的数据，将数据转化为四个维度的数据，通道优先[-1,3(通道),32,32]
imgs = np.reshape(data, [-1, 3, 32, 32])	

#for循环对所有图片进行遍历
for i in range(imgs.shape[0]):
    im_data = imgs[i, ...]
	#对图像数据的通道维度进行交换，原始的3调到最后，即[0,1,2] -> [1,2,0]
    im_data = np.transpose(im_data, [1, 2, 0])
	#转化数据格式 RGB -> BGR	
    im_data = cv2.cvtColor(im_data, cv2.COLOR_RGB2BGR)
	#定义数据存放路径
    f = "{}/{}".format("data/image/train", classification[labels[i]])	
	
	#判断文件夹路径是否存在
    if not os.path.exists(f):	
        #不存在就创建
		os.mkdir(f)
	#向指定的文件夹写入图片
    cv2.imwrite("{}/{}.jpg".format(f, str(i)), im_data)













