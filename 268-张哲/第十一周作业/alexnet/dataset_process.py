import os
photos = os.listdir("./data/image/train/")
#生成训练集txt文件
with open("./data/dataset.txt","w") as f:
    for photo in photos:
        name = photo.split(".")[0]
        if name == 'cat':
            f.write(photo + ";0\n")
        elif name=="dag":
            f.write(photo + ";1\n")
f.close()