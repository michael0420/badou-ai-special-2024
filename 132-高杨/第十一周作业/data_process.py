import os

photos = os.listdir('data/train')

# 这部分用于生成标签数据 ， 把所有的图片的标签都进行
with open('data/dataset2.txt','w') as f:
    for photo in photos:
        name = photos.split('.')[0]
        if name=='cat':
            f.write(photo + ';0\n')
        elif name=='dog':
            f.write(photo + ';1\n')
f.close()


