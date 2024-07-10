'''
该部分用于读取数据图像的名称，并根据名称制作对应标签，写入数据集
'''

import os

photos = os.listdir('data/image/train/')

with open('data/dataset.txt', 'w') as f:
    for photo in photos:
        name = photo.split('.')[0]
        if name == 'cat':
            f.write(f'{photo};0\n')
        elif name == 'dog':
            f.write(f'{photo};1\n')
f.close()
