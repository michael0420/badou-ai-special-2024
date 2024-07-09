import os

photos = os.listdir("../../../../../data/AlexNet-Keras-master/train/")

with open('../../../../../data/AlexNet-Keras-master/dataset.txt', 'w') as f:
    for photo in photos:
        name = photo.split(".")[0]
        if name == 'cat':
            f.write(photo + ";0\n")
        elif name == "dog":
            f.write(photo + ";1\n")
f.close()