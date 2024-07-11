import os

# 图片保存的地址
photos = os.listdir("./data/image/train/")

# 保存的数据集在txt上 模式为写
with open("data/dataset.txt", "w") as f:
    for photo in photos:
        # 切割得到字符串为.之前的为名字
        name = photo.split(".")[0]
        # 猫的标签为0
        if name == "cat":
            f.write(photo + ";0\n")
        # 狗的标签为1
        elif name == "dog":
            f.write(photo + ";1\n")
f.close()