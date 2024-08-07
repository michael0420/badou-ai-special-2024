# 这个文件用于处理数据，生成带标签的dataset文件
import os

# 列出"./data/image/train/"目录下的所有文件和文件夹
photos = os.listdir("./data/image/train/")

# 该部分用于将
with open("data/dataset.txt", "w") as f:
    for photo in photos:
        name = photo.split(".")[0]
        if name == "cat":
            f.write(photo + ";0\n")
        elif name == "dog":
            f.write(photo + ";1\n")
