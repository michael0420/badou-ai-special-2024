from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K

# # K.set_image_dim_ordering('tf')
# K.set_image_data_format('channels_last')
#
#
# def generate_arrays_from_file(lines, batch_size):
#     # 获取总长度
#     n = len(lines)
#     i = 0
#     while 1:
#         X_train = []
#         Y_train = []
#         # 获取一个batch_size大小的数据
#         for b in range(batch_size):
#             if i == 0:
#                 np.random.shuffle(lines)
#             name = lines[i].split(';')[0]
#             # 从文件中读取图像
#             img = cv2.imread(r".\data\image\train" + '/' + name)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = img / 255
#             X_train.append(img)
#             Y_train.append(lines[i].split(';')[1])
#             # 读完一个周期后重新开始
#             i = (i + 1) % n
#         # 处理图像
#         X_train = utils.resize_image(X_train, (224, 224))
#         X_train = X_train.reshape(-1, 224, 224, 3)
#         Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)
#         yield (X_train, Y_train)
#
#
# if __name__ == "__main__":
#     # 模型保存的位置
#     log_dir = "./logs/"
#
#     # 打开数据集的txt
#     with open(r".\data\dataset.txt", "r") as f:
#         lines = f.readlines()
#
#     # 打乱行，这个txt主要用于帮助读取数据来训练
#     # 打乱的数据更有利于训练
#     np.random.seed(10101)
#     np.random.shuffle(lines)
#     np.random.seed(None)
#
#     # 90%用于训练，10%用于估计。
#     num_val = int(len(lines) * 0.1)
#     num_train = len(lines) - num_val
#
#     # 建立AlexNet模型
#     model = AlexNet()
#
#     # 保存的方式，3代保存一次
#     checkpoint_period1 = ModelCheckpoint(
#         log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
#         monitor='acc',
#         save_weights_only=False,
#         save_best_only=True,
#         period=3
#     )
#     # 学习率下降的方式，acc三次不下降就下降学习率继续训练
#     reduce_lr = ReduceLROnPlateau(
#         monitor='acc',
#         factor=0.5,
#         patience=3,
#         verbose=1
#     )
#     # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
#     early_stopping = EarlyStopping(
#         monitor='val_loss',
#         min_delta=0,
#         patience=10,
#         verbose=1
#     )
#
#     # 交叉熵
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=Adam(lr=1e-3),
#                   metrics=['accuracy'])
#
#     # 一次的训练集大小
#     batch_size = 128
#
#     print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
#
#     # 开始训练
#     model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
#                         steps_per_epoch=max(1, num_train // batch_size),
#                         validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
#                         validation_steps=max(1, num_val // batch_size),
#                         epochs=50,
#                         initial_epoch=0,
#                         callbacks=[checkpoint_period1, reduce_lr])
#     model.save_weights(log_dir + 'last1.h5')


from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K

K.set_image_data_format('channels_last')


# # 第一次
# def generate_arrays_from_file(lines, batch_size):
#     # 获取总长度
#     n = len(lines)
#     print(n)
#     i = 0
#     while 1:
#         X_train = []
#         Y_train = []
#
#         # 获取一个batch_size大小的数据
#         for b in range(batch_size):
#             if i == 0:
#                 np.random.shuffle(lines)
#
#             name = lines[i].split(';')[0]
#             # 从文件中读取图像
#             img = cv2.imread("./data/image/train/" + name)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = img / 255
#             X_train.append(img)
#             Y_train.append(lines[i].split(';')[1])
#             i = (i + 1) % n
#         X_train = utils.resize_image(X_train, (224, 224))
#         X_train = X_train.reshape(-1, 224, 224, 3)
#         Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)
#         yield X_train, Y_train
#
#
# if __name__ == '__main__':
#     # 模型保存的位置
#     log_dir = './logs/'
#
#     # 读取dataset数据集
#     # 路径分隔符bug？
#     with open(r".\data\dataset.txt","r") as f:
#         lines = f.readlines()
#     print(len(lines))
#
#     # 打乱lines列表中的每个数据
#     np.random.seed(10101)
#     np.random.shuffle(lines)
#     np.random.seed(None)
#
#     # 90%的数据用来训练，10%的数据用于推理测试
#     num_val = int(len(lines) * 0.1)
#     num_train = len(lines) - num_val
#
#     # 构建网络模型
#     model = AlexNet()
#
#     # 保存方式，3代保存一次
#     checkpoint_period1 = ModelCheckpoint(
#         log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-.h5',
#         monitor='acc', save_weights_only=False, save_best_only=True, period=3)
#
#     # 学习率下降的方式
#     reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1)
#
#     # 是否需要早停
#     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
#
#     # 交叉熵
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=Adam(lr=1e-3),
#                   metrics=['accuracy'])
#
#     batch_size = 128
#     print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
#
#     # 开始训练
#     model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
#                         steps_per_epoch=max(1, num_train // batch_size),
#                         validation_data=generate_arrays_from_file(lines[num_train:],batch_size),
#                         validation_steps=max(1, num_val // batch_size),
#                         epochs=50,
#                         initial_epoch=0,
#                         callbacks=[checkpoint_period1, reduce_lr])
#     model.save_weights(log_dir + 'last1.h5')
#


# #  第一次
# def generate_arrays_from_file(lines, batch_size):
#     n = len(lines)
#     i = 0
#     while 1:
#         train_data = []
#         train_label = []
#         for b in range(batch_size):
#             if i == 0:
#                 np.random.shuffle(lines)
#             img_name = lines[i].split(';')[0]
#             img_name = './data/image/train/' + img_name
#             img = cv2.imread(img_name)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img_data = img / 255
#             img_label = lines[i].split(';')[1]
#
#             train_data.append(img_data)
#             train_label.append(img_label)
#
#             i = (i + 1) % n
#
#         train_data = utils.resize_image(train_data, (224, 224))
#         train_data = train_data.reshape((-1, 224, 224, 3))
#         train_label = np_utils.to_categorical(np.array(train_label), num_classes=2)
#         yield train_data, train_label
#
#     pass
#
#
# if __name__ == '__main__':
#     # 模型保存的路径
#     log_dir = './logs/'
#     # 读取数据集
#     with open('./data/dataset.txt', 'r') as f:
#         lines = f.readlines()
#     # 打乱行
#     np.random.seed(10101)
#     np.random.shuffle(lines)
#     np.random.seed(None)
#     # 90%用于训练，10%用于检测
#     num_test = int(len(lines) * 0.1)
#     num_train = len(lines) - num_test
#     # 构建模型
#     model = AlexNet()
#     # 保存的方式，3代保存一次
#     checkpoint_period1 = ModelCheckpoint('./logs/', monitor='acc', verbose=1,
#                                          save_best_only=True, save_weights_only=False, period=3)
#     # 学习率下降的方式，acc三次不下降就下降学习率继续训练
#     reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1)
#     # 是否需要早停，当val_loss一直不下降就可以停止
#     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
#     # 设置损失函数为交叉熵
#     model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
#     # 一次训练集的大小
#     batch_size = 128
#     # 真正读取数据
#     train_generator = generate_arrays_from_file(lines[:num_train], batch_size=128)
#     test_generator = generate_arrays_from_file(lines[num_train:len(lines)], batch_size=128)
#     # 训练
#     model.fit(train_generator,
#               steps_per_epoch=max(1, num_train // batch_size),
#               epochs=50,
#               verbose=1,
#               callbacks=[checkpoint_period1, reduce_lr],
#               validation_data=test_generator,
#               validation_steps=max(1, num_test // batch_size),
#               )
#     # 保存模型
#     model.save_weights(log_dir + 'last1.h5')


# #  第二次
# def generate_arrays_from_file(lines, batch_size):
#     i = 0
#     n = len(lines)
#     while True:
#         train_data = []
#         train_label = []
#         for b in range(batch_size):
#             if i == 0:
#                 np.random.shuffle(lines)
#             img_name = lines[i].split(';')[0]
#             img_label = lines[i].split(';')[1]
#             img_data = cv2.imread('xxxxxx' + '/' + img_name)
#             img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
#             img_data = img_data / 255
#             train_data.append(img_data)
#             train_label.append(img_label)
#             i = (i + 1) % n
#         train_data = utils.resize_image(train_data, (224, 224))
#         train_data = train_data.reshape((-1, 224, 224, 3))
#         train_label = np_utils.to_categorical(np.array(train_label), num_classes=2)
#         yield train_data, train_label
#     pass
#
#
# if __name__ == '__main__':
#     # 模型保存的路径
#     log_dir = './xxxxx'
#     # 读取训练数据
#     with open('./xxxxxx', 'r') as f:
#         lines = f.readlines()
#     # 随机打乱
#     np.random.seed(10101)
#     np.random.shuffle(lines)
#     np.random.seed(None)
#     # 90%用于训练，10%用于检测
#     num_test = int(len(lines) * 0.1)
#     num_train = len(lines) - num_test
#     # 构建模型
#     model = AlexNet()
#     # 保存方式
#     checkpoint_period1 = ModelCheckpoint(filepath='xxxxxxxxxxx', monitor='acc', verbose=1,
#                                          save_best_only=True, save_weights_only=False, period=3)
#     # 学习率
#     reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=10, verbose=1)
#     # 是否需要早停
#     EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1)
#     # 交叉熵
#     model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
#     # 真正读取数据
#     train_generator = generate_arrays_from_file(lines[:num_train], batch_size=128)
#     test_generator = generate_arrays_from_file(lines[num_train:len(lines)], batch_size=128)
#     # 训练
#     model.fit(train_generator,
#               steps_per_epoch=max(1, num_train // batch_size),
#               epochs=50,
#               verbose=1,
#               callbacks=[checkpoint_period1, reduce_lr],
#               validation_data=test_generator,
#               validation_steps=max(1, num_test // batch_size),
#               )
#     # 保存模型
#     # xxxx表示路径
#     model.save_weights(log_dir + 'xxxxxxx')
#     pass


# #  第三次
# def generate_arrays_from_file(lines, batch_size):
#     n = len(lines)
#     i = 0
#     while True:
#         train_data = []
#         train_label = []
#         for b in range(batch_size):
#             if i == 0:
#                 np.random.shuffle(lines)
#             img_name = lines[i].split(';')[0]
#             img = cv2.imread('./xxxx' + '/' + img_name)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img_data = img / 255
#             img_label = lines[i].split(';')[1]
#             train_data.append(img_data)
#             train_label.append(img_label)
#             i = (i + 1) % n
#         train_data = utils.resize_image(train_data, (224, 224))
#         train_data = train_data.reshape((-1, 224, 224, 3))
#         train_label = np_utils.to_categorical(np.array(train_label), num_classes=2)
#         yield train_data, train_label
#
#
# if __name__ == '__main__':
#     # 读取训练数据
#     with open('xxxxxxxxxxxxxx', 'r') as f:
#         lines = f.readlines()
#     # 进行打乱
#     np.random.seed(10101)
#     np.random.shuffle(lines)
#     np.random.seed(None)
#     # 90%用于训练，10%用于检测
#     num_test = int(len(lines) * 0.1)
#     num_train = len(lines) - num_test
#     # 构建模型
#     model = AlexNet()
#     # 训练的中间结果保存方式
#     checkpoint_period1 = ModelCheckpoint('./xxxxxxxxxxx', monitor='acc', verbose=1,
#                                          save_best_only=True, save_weights_only=False, period=3)
#     # 学习率
#     reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1)
#     # 早停，防止过拟合
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
#     # 添加一些优化参数，损失参数，交叉熵
#     model.compile(optimizer=Adam(lr=1e-3), loss='categotical_crossentropy', metrics=['accuracy'])
#     # 真正读取数据
#     train_generator = generate_arrays_from_file(lines[:num_train], batch_size=128)
#     test_generator = generate_arrays_from_file(lines[num_train:len(lines)], batch_size=128)
#     # 训练
#     model.fit(train_generator,
#               steps_per_epoch=max(1, num_train // batch_size),
#               epochs=50,
#               verbose=1,
#               callbacks=[checkpoint_period1, reduce_lr],
#               validation_data=test_generator,
#               validation_steps=max(1, num_test // batch_size),
#               )
#     # 保存训练好的模型
#     model.save_weights('./xxxxxxxxxxxxxxxxxxxx')
#     pass


# #  第四次
# def generate_arrays_from_file(lines, batch_size):
#     i = 0
#     n = len(lines)
#     while True:
#         train_data = []
#         train_label = []
#         for b in range(batch_size):
#             if i == 0:
#                 np.random.shuffle(lines)
#             img_name = lines[i].split(';')[0]
#             img_data = cv2.imread('./xxxxxxxxx' + '/' + img_name)
#             img_data = cv2.cvtColor((img_data, cv2.COLOR_BGR2RGB))
#             img_data = img_data / 255
#             img_label = lines[i].split(';')[1]
#             train_data.append(img_data)
#             train_label.append(img_label)
#             i = (i + 1) % n
#         train_data = utils.resize_image(train_data, (224, 224))
#         train_data = train_data.reshape((-1, 224, 224, 3))
#         train_label = np_utils.to_categorical(np.array(train_label), num_classes=2)
#         yield train_data, train_label
#
#
# if __name__ == '__main__':
#     # 读取数据
#     with open('./xxxxxxxxxxxxx', 'r') as f:
#         lines = f.readlines()
#     # 打乱数据
#     np.random.seed(10101)
#     np.random.shuffle(lines)
#     np.random.seed(None)
#     # 90%用于训练，10%用于检测
#     num_test = int(len(lines) * 0.1)
#     num_train = len(lines) - num_test
#     # 构建模型
#     model = AlexNet()
#     # 保存方式
#     checkpoint_period1 = ModelCheckpoint('./xxxxxxxxxx', monitor='acc', verbose=1,
#                                          save_weights_only=False,
#                                          save_best_only=True, period=3)
#     # 学习率
#     reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1)
#     # 早停
#     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
#     # 交叉熵
#     model.compile(optimizer=Adam(lr=1e-3), loss='categotical_crossentropy', metrics=['accuracy'])
#
#     # 真正读取数据
#     train_generator = generate_arrays_from_file(lines[:num_train], batch_size=128)
#     test_generator = generate_arrays_from_file(lines[num_train:len(lines)], batch_size=128)
#     # 训练
#     model.fit(train_generator,
#               steps_per_epoch=max(1, num_train // batch_size),
#               epochs=50,
#               verbose=1,
#               callbacks=[checkpoint_period1, reduce_lr],
#               validation_data=test_generator,
#               validation_steps=max(1, num_test // batch_size),
#               )
#     # 保存模型
#     model.save_weights('./xxxxxxxxxxxxxxxxx')
#

# # 第五次
# def generate_arrays_from_file(lines, batch_size):
#     i = 0
#     n = len(lines)
#     while True:
#         train_data = []
#         train_label = []
#         for i in range(batch_size):
#             if i == n:
#                 np.random.shuffle(lines)
#             img_name = lines[i].split(';')[0]
#             img_data = cv2.imread('./xxxxxxxxxxx' + '/' + img_name)
#             img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
#             img_data = img_data / 255
#             img_label = lines[i].split(';')[1]
#             train_data.append(img_data)
#             train_label.append(img_label)
#             i = (i + 1) % n
#         train_data = utils.resize_image(train_data, (224, 224))
#         train_data = train_data.reshape((-1, 224, 224, 3))
#         train_label = np_utils.to_categorical(np.array(train_label), num_classes=2)
#         yield train_data, train_label
#
#
# if __name__ == '__main__':
#     # 创建模型
#     model = AlexNet()
#     # 读取图片的文件名
#     with open('./xxxxxxxxxxxx', 'r') as f:
#         lines = f.readlines()
#     # 打乱数据
#     np.random.seed(10101)
#     np.random.shuffle(lines)
#     np.random.seed(None)
#     # 90%用于训练，10%用于检测
#     num_test = int(len(lines) * 0.1)
#     num_train = len(lines) - num_test
#     # 保存方式
#     checkpoint_period1 = ModelCheckpoint('./xxxxxxxx', monitor='acc', verbose=1,
#                                          save_best_only=True, save_weights_only=False,
#                                          period=3)
#     # 学习率
#     reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1)
#     # 早停
#     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
#     # 交叉熵
#     model.compile(optimizer=Adam(1e-3), loss='categotical1_crossentropy', metrics=['accuracy'])
#     # 真正读取数据
#     train_generator = generate_arrays_from_file(lines[:num_train], batch_size=128)
#     test_generator = generate_arrays_from_file(lines[num_train:len(lines)], batch_size=128)
#     # 训练
#     model.fit(train_generator,
#               steps_per_epoch=max(1, num_train // batch_size),
#               epochs=50,
#               verbose=1,
#               callbacks=[checkpoint_period1, reduce_lr],
#               validation_data=test_generator,
#               validation_steps=max(1, num_test // batch_size),
#               )
#     # 保存模型
#     model.save_weights('./xxxxxxxxxxx')
#

# # 第六次
# def generate_arrays_from_file(lines, batch_size):
#     i = 0
#     n = len(lines)
#     while True:
#         train_data = []
#         train_label = []
#         for b in range(batch_size):
#             if i == 0:
#                 np.random.shuffle(lines)
#             img_name = lines[i].split(';')[0]
#             img_data = cv2.imread('./xxxxxxxxx' + '/' + img_name)
#             img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
#             img_data = img_data / 255
#             img_label = lines[i].split(';')[1]
#             train_data.append(img_data)
#             train_label.append(img_label)
#             i = (i + 1) % n
#         train_data = utils.resize_image(train_data, (224, 224))
#         train_data = train_data.reshape((-1, 224, 224, 3))
#         train_label = np_utils.to_categorical(np.array(train_label), num_classes=2)
#         yield (train_data, train_label)
#
#
# if __name__ == '__main__':
#     # 创建模型
#     model = AlexNet()
#     # 加载图片的文件名
#     with open('./xxxxxxxxx', 'r') as f:
#         lines = f.readlines()
#     # 打乱数据
#     np.random.seed(10101)
#     np.random.shuffle(lines)
#     np.random.seed(None)
#     # 90%用于训练，10%用于检测
#     num_test = int(len(lines) * 0.1)
#     num_train = len(lines) - num_test
#     # 保存方式
#     checkpoint_period1 = ModelCheckpoint(filepath='./xxxxxxxxxx', monitor='acc', verbose=1,
#                                          save_weights_only=False, save_best_only=True, period=3)
#     # 学习率
#     reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=10, verbose=1)
#     # 早停
#     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
#     # 交叉熵
#     model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
#
#     # 真正读取数据
#     train_generator = generate_arrays_from_file(lines[:num_train], batch_size=128)
#     test_generator = generate_arrays_from_file(lines[num_train:len(lines)], batch_size=128)
#     # 训练
#     model.fit(train_generator,
#               steps_per_epoch=max(1, num_train // batch_size),
#               epochs=50,
#               verbose=1,
#               callbacks=[checkpoint_period1, reduce_lr],
#               validation_data=test_generator,
#               validation_steps=max(1, num_test // batch_size),
#               )
#     # model.fit_generator(train_generator,
#     #                     steps_per_epoch=max(1, num_train // batch_size),
#     #                     epochs=50,
#     #                     verbose=1,
#     #                     callbacks=[checkpoint_period1, reduce_lr],
#     #                     validation_data=test_generator,
#     #                     validation_steps=max(1, num_test//batch_size),
#     #                     initial_epoch=0)
#
#     # 保存训练完成的模型
#     model.save_weights('./xxxx')


# # 第七次
# def generate_arrays_from_file(lines,batch_size):
#     i=0
#     n=len(lines)
#     while True:
#         train_data=[]
#         train_label=[]
#         for b in range(batch_size):
#             if i==0:
#                 np.random.shuffle(lines)
#             img_name=lines[i].split(';')[0]
#             img_label=lines[i].split(';')[1]
#             img_data=cv2.imread('./xxx'+'/'+img_name)
#             img_data=cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
#             img_data=img_data/255
#             train_data.append(img_data)
#             train_label.append(img_label)
#             i=(i+1)%n
#         train_data=utils.resize_image(train_data,(224,224))
#         train_data=train_data.reshape((-1,224,224,3))
#         train_label=np_utils.to_categorical(np.array(train_label),num_classes=2)
#         yield (train_data,train_label)
#
# if __name__=='__main__':
#     # 构建模型
#     model=AlexNet()
#     # 读取数据的文件名
#     with open("./xxxxxxxxxxx",'r') as f:
#         lines=f.readlines()
#     # 打乱数据
#     np.random.seed(10101)
#     np.random.shuffle(lines)
#     np.random.seed(None)
#     # 90%用于训练，10%用于中途测试
#     num_test=int(len(lines)*0.1)
#     num_train=len(lines)-num_test
#     # 保存方式
#     checkpoint_period1=ModelCheckpoint('./xxxxxxxx',monitor='acc',verbose=1,
#                                        save_best_only=True,save_weights_only=False,period=3)
#     # 学习率
#     reduce_lr=ReduceLROnPlateau(monitor='acc',factor=0.5,patience=3,verbose=1)
#     # 早停
#     early_stopping=EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=1)
#     # 交叉熵
#     model.compile(optimizer=Adam(1e-3),loss='categorical_crossentropy',metrics=['accuracy'])
#     # 真正读取文件
#     batch_size=128
#     train_generator=generate_arrays_from_file(lines[:num_train],batch_size=batch_size)
#     test_generator=generate_arrays_from_file(lines[num_train:],batch_size=batch_size)
#     # 训练
#     model.fit_generator(train_generator,
#                         steps_per_epoch=max(1,num_train//batch_size),
#                         epochs=50,verbose=1,
#                         callbacks=[checkpoint_period1,reduce_lr,early_stopping],
#                         validation_data=test_generator,
#                         validation_steps=max(1,num_test//batch_size)
#                         )
#     model.save_weights('./xxxxxx')
#


# # 第八次
# def generate_arrays_from_file(lines, batch_size):
#     i = 0
#     n = len(lines)
#     while True:
#         datas = []
#         labels = []
#         for j in range(batch_size):
#             if i == 0:
#                 np.random.shuffle(lines)
#             image_name = lines[i].split(';')[0]
#             image_label = lines[i].split(';')[1]
#             image_data = cv2.imread('./xxxx' + '/' + image_name)
#             image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
#             image_data = image_data / 255
#             datas.append(image_data)
#             labels.append(image_label)
#             i = (i + 1) % n
#         datas = utils.resize_image(datas, (224, 224))
#         datas = datas.reshape((-1, 224, 224, 3))
#         labels = np_utils.to_categorical(np.array(labels), num_classes=2)
#         yield (datas, labels)
#
#
# if __name__ == '__main__':
#     model = AlexNet()
#     # 读取图片的名称
#     with open('./xxxxxxxxxxx', 'r') as f:
#         lines = f.readlines()
#     # 打乱数据
#     np.random.seed(10101)
#     np.random.shuffle(lines)
#     np.random.seed(None)
#     # 90%用于训练，10%用于中期测试
#     num_test = int(len(lines) * 0.1)
#     num_train = len(lines) - num_test
#     # 保存方式
#     checkpoint_period1 = ModelCheckpoint(filepath='./xxx', monitor='acc', verbose=1,
#                                          save_weights_only=False, save_best_only=True, period=3)
#     # 学习率
#     reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1)
#     # 早停
#     early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1)
#     # 交叉熵
#     model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
#     # 真正读取数据
#     train_generator = generate_arrays_from_file(lines[:num_train], batch_size=128)
#     test_generator = generate_arrays_from_file(lines[num_train:], batch_size=128)
#     batch_size = 128
#     # 训练
#     model.fit_generator(train_generator,
#                         steps_per_epoch=max(1, num_train // batch_size),
#                         epochs=50,
#                         verbose=1,
#                         callbacks=[checkpoint_period1, reduce_lr, early_stopping],
#                         validation_data=test_generator,
#                         validation_steps=max(1, num_test // batch_size))
#     # 保存模型
#     model.save_weights('./xxxxxx')
#
#
# # 第九次
# def generate_arrays_from_file(lines, batch_size):
#     i = 0
#     n = len(lines)
#     while True:
#         train_data = []
#         train_label = []
#         for j in range(batch_size):
#             if i == 0:
#                 np.random.shuffle(lines)
#             img_name = lines[i].split(';')[0]
#             img_label = lines[i].split(';')[1]
#             img_data = cv2.imread('./xx' + '/' + img_name)
#             img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
#             img_data = img_data / 255
#             train_data.append(img_data)
#             train_label.append(img_label)
#             i = (i + 1) % n
#         train_data = utils.resize_image(train_data, (224, 224))
#         train_data = train_data.reshape((-1, 224, 224, 3))
#         train_label = np_utils.to_categorical(np.array(train_label), num_classes=2)
#         yield (train_data, train_label)
#
#
# if __name__ == '__main__':
#     # 构建模型
#     model = AlexNet()
#     # 读取图片的文件名数据到一个列表中
#     with open('./xxxxxxx', 'r') as f:
#         lines = f.readlines()
#     # 打乱数据
#     np.random.seed(10101)
#     np.random.shuffle(lines)
#     np.random.seed(None)
#     # 90%的数据用于训练，10%的数据用于测试
#     num_test = int(len(lines) * 0.1)
#     num_train = len(lines) - num_test
#     # 设置一些参数
#     # 保存方式
#     checkpoint_period1 = ModelCheckpoint(filepath='./xxx', monitor='acc', verbose=1,
#                                          save_best_only=True, save_weights_only=False, period=3)
#     # 学习率
#     reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1)
#     # 早停
#     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
#     # 损失函数交叉熵
#     model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
#
#     # 真正读取数据
#     batch_size = 128
#     train_generator = generate_arrays_from_file(lines[:num_train], batch_size=batch_size)
#     test_generator = generate_arrays_from_file(lines[num_train:], batch_size=batch_size)
#     # 训练
#     model.fit_generator(train_generator,
#                         steps_per_epoch=max(1, num_train // batch_size),
#                         epochs=50, verbose=1,
#                         callbacks=[checkpoint_period1, reduce_lr, early_stopping],
#                         validation_data=test_generator,
#                         validation_steps=max(1, num_test // batch_size)
#                         )
#     model.save_weights('./xxxxxxx')
#
#
# # 第十次
# def generate_arrays_from_file(lines, batch_size):
#     i = 0
#     n = len(lines)
#     while True:
#         train_data = []
#         train_label = []
#         for j in range(batch_size):
#             if i == 0:
#                 np.random.shuffle(lines)
#             img_name = lines[i].split(';')[0]
#             img_label = lines[i].split(';')[1]
#             img_data = cv2.imread('./xxx' + '/' + img_name)
#             img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
#             img_data = img_data / 255
#             train_data.append(img_data)
#             train_label.append(train_label)
#             i = (i + 1) % n
#         train_data = utils.resize_image(train_data, (224, 224))
#         train_data = train_data.reshape((-1, 224, 224, 3))
#         train_label = np_utils.to_categorical(np.array(train_label), num_classes=2)
#         yield (train_data, train_label)
#
#
# if __name__ == '__main__':
#     # 读取图片的名称
#     with open('./xxx', 'r') as f:
#         lines = f.readlines()
#     # 打乱数据
#     np.random.seed(10101)
#     np.random.shuffle(lines)
#     np.random.seed(None)
#     # 90%用于训练，10%用于测试
#     num_test = int(len(lines) * 0.1)
#     num_train = len(lines) - num_test
#     # 构建模型
#     model = AlexNet()
#     # 给模型添加一些参数
#     # 保存方式
#     checkpoint_period1 = ModelCheckpoint(filepath='./xx', monitor='acc', verbose=1,
#                                          save_weights_only=False, save_best_only=True, period=3)
#     # 学习率
#     reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1)
#     # 早停
#     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)
#     # 损失函数为交叉熵
#     model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
#     batch_size = 128
#     # 真正读取数据
#     train_generator = generate_arrays_from_file(lines[:num_train])
#     test_generator = generate_arrays_from_file(lines[num_train:])
#     # 训练
#     model.fit_generator(train_generator,
#                         steps_per_epoch=max(1, num_train // batch_size),
#                         epochs=50, verbose=1,
#                         validation_data=test_generator,
#                         validation_steps=max(1, num_test // batch_size),
#                         callbacks=[checkpoint_period1, reduce_lr, early_stopping])
#     model.save_weights('./xxxx')
#
#
# # 第十一次
# def generate_arrays_from_file(lines,batch_size):
#     i=0
#     n=len(lines)
#     while True:
#         train_data=[]
#         train_label=[]
#         for j in range(batch_size):
#             if i==0:
#                 np.random.shuffle(lines)
#             img_name=lines[i].split(';')[0]
#             img_label=lines[i].split(';')[1]
#
#             img_data=cv2.imread('./xxx'+'/'+img_name)
#             img_data=cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
#             img_data=img_data/255
#
#             train_data.append(img_data)
#             train_label.append(img_label)
#             i=(i+1)%n
#         train_data=utils.resize_image(train_data,(224,224))
#         train_data=train_data.reshape((-1,224,224,3))
#         train_label=np_utils.to_categorical(np.array(train_label),num_classes=2)
#         yield train_data,train_label
#     pass
#
# if __name__=='__main__':
#     # 读取图片的文件名
#     with open('./xxx','r') as f:
#         lines=f.readlines()
#     # 打乱
#     np.random.seed(10010)
#     np.random.shuffle(lines)
#     np.random.seed(None)
#     # 90%用于训练，10%用于测试
#     num_train=int(len(lines)*0.9)
#     num_test=len(lines)-num_train
#     # 加载模型
#     model=AlexNet()
#     # 添加一些参数
#     # 权重的保存方式
#     checkpoint_period1=ModelCheckpoint(filepath='./xxx',monitor='acc',verbose=1,
#                                        save_best_only=True,save_weights_only=False,period=3)
#     # 学习率
#     reduce_lr=ReduceLROnPlateau(monitor='acc',factor=0.5,patience=3,verbose=1)
#     # 早停
#     early_stopping=EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=1)
#     # 交叉熵损失函数
#     model.compile(optimizer=Adam(1e-3),loss='categorical_crossentropy',metrics=['accuracy'])
#     # 真正读取训练和测试数据
#     batch_size=128
#     train_generator=generate_arrays_from_file(lines[:num_train],batch_size=batch_size)
#     test_generator=generate_arrays_from_file(lines[num_train:],batch_size=batch_size)
#     # 训练
#     model.fit_generator(train_generator,
#                         steps_per_epoch=max(1,num_train//batch_size),
#                         epochs=50,verbose=1,
#                         validation_data=test_generator,
#                         validation_steps=max(1,num_test//batch_size),
#                         callbacks=[checkpoint_period1,reduce_lr,early_stopping])
#     model.save_weights('./xxxx')


# 第十二次
def generate_arrays_from_file(lines, batch_size):
    i = 0
    n = len(lines)
    while True:
        data = []
        label = []
        for k in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            img_name = lines[i].split(';')[0]
            img_label = lines[i].split(';')[1]

            # 读取数据
            img_data = cv2.imread('xxxx' + '/' + img_name)
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            img_data = img_data / 255

            data.append(img_data)
            label.append(img_label)
            i = (i + 1) % n

        data = utils.resize_image(data, size=(224, 224))
        data = data.reshape([-1, 224 * 224 * 3])
        label = np_utils.to_categorical(np.array(label), num_classes=2)
        yield data, label


if __name__ == '__main__':
    # 构建模型
    model = AlexNet()
    # 准备数据
    with open('./xxxxx', 'r') as f:
        lines = f.readlines()
    # 打乱数据
    np.random.seed(10010)
    np.random.shuffle(lines)
    np.random.seed(None)
    # 90%用于训练，10%用于检测
    num_train = int(len(lines) * 0.9)
    num_test = len(lines) - num_train
    # 设置一些参数
    # 保存方式
    checkpoint_period1 = ModelCheckpoint('./xxxx', monitor='acc', verbose=3,
                                         save_weights_only=False, save_best_only=True,
                                         period=3)
    # 学习率下降
    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=3, verbose=1)
    # 早停
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    # 编译
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    #
    # 真正的读取数据
    batch_size = 128
    train_generator = generate_arrays_from_file(lines[:num_train], batch_size=batch_size)
    test_generator = generate_arrays_from_file(lines[num_train:], batch_size=batch_size)
    model.fit_generator(train_generator,
                        steps_per_epoch=max(1, num_train // batch_size),
                        epochs=5, verbose=1, callbacks=[checkpoint_period1, reduce_lr, early_stopping],
                        validation_data=test_generator,
                        validation_steps=max(1, num_test // batch_size))
