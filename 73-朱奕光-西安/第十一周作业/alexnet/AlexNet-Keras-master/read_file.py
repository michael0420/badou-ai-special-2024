import cv2
import numpy as np
from utils_homework import img_resize
import keras

def read_file(data,batch_size):
    n = len(data)
    i = 0
    while True:  #没有明确的终止条件，根据外部调用条件终止（epoch）
        X_train = []
        Y_train = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(data)
            name = data[i].split(';')[0]
            img = cv2.imread(r".\data\image\train" + '/' + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img/255
            X_train.append(img)
            Y_train.append(data[i].split(';')[1])
            i = (i+1) % n
        X_train = img_resize(X_train,(224,224))
        X_train = X_train.reshape(-1,224,224,3)
        Y_train = keras.utils.to_categorical(np.array(Y_train),num_classes= 2)
        yield(X_train, Y_train)