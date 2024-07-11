import cv2
import numpy as np

def img_resize(img,size):
    image = []
    for i in img:
        img = cv2.resize(i, size)
        image.append(img)
    image = np.asarray(image)
    return image

def answer(perdict_res):
    bingo = []
    with open("./data/model/index_word.txt", "r", encoding='utf-8') as f:
        for line in f:
            a = line.split(';')[1][:-1]
            bingo.append(a)
    return bingo[perdict_res]

