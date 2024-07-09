from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os

def display_imgs(imgs, titles, rows, cols):
    for i in range( len(imgs) ):
        plt.subplot(rows, cols, i+1)
        plt.imshow(imgs[i], cmap="gray")
        plt.title(titles[i])
        plt.xticks([]) , plt.yticks([]) # 不显示横纵坐标轴
    plt.show()

if __name__ == "__main__":
    # 加载测试数据
    (imgs_train, labels_train) , (imgs_test, labels_test) = mnist.load_data()

    # 将2维图片数据调整为1维,并做归一化处理
    num_pixels = imgs_train.shape[1] * imgs_train.shape[2]
    x_train = imgs_train.reshape(imgs_train.shape[0], num_pixels).astype('float32') / 255.0
    x_test = imgs_test.reshape(imgs_test.shape[0], num_pixels).astype('float32') / 255.0

    # 将labels转化为one-hot编码
    y_train = np_utils.to_categorical(labels_train)
    y_test = np_utils.to_categorical(labels_test)
    num_classes = y_train.shape[1]

    # 定义模型
    model = Sequential()
    model.add(Dense(units=16, activation='relu', input_dim=num_pixels)) # 输入层
    # model.add(Dense(units=64, activation='relu'))# 中间层
    model.add(Dense(units=num_classes, activation='softmax')) #输出层

    # 编译模型
    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy'])

    # 训练模型
    model.fit(x=x_train, y=y_train, batch_size=600, epochs=10)

    # 评估模型
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"loss:{loss}, accuracy:{accuracy}")

    # 随机抽取100张测试图片，测试预测效果
    num_choice = 100
    indices = np.random.choice(imgs_test.shape[0], size=num_choice, replace=False)
    imgs_choice = imgs_test[indices]
    x_choice = x_test[indices]

    # 进行预测
    y_pred = model.predict(x_choice)

    # 获取预测的标签
    labels_pred = y_pred.argmax(axis=1)

    # 展示预测结果，每张图片的标题即为预测结果
    display_imgs(
        imgs=imgs_choice,
        titles=labels_pred,
        rows=5, cols=20
    )

    # 保存模型
    os.system('mkdir -p saved_model')
    model.save('saved_model/my_model')
