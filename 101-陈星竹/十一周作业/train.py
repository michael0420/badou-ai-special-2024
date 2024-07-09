from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import pic_cut
import cv2
from keras import backend as K

#配置Keras的图像维度顺序为TensorFlow('tf')
K.set_image_data_format('channels_last')

#数据生成器函数
def generate_arrays(lines,batch_size):
    n = len(lines)
    i = 0
    while 1 :
        x_train = []
        y_train = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines) #随机打乱
            name = lines[i].split(";")[0]
            img = cv2.imread(r".\data\image\train" + '/' + name)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img / 255
            x_train.append(img)
            y_train.append(lines[i].split(";")[1])
            i = (i+1) % n

        x_train = pic_cut.resize_img(x_train,(224,224))
        x_train = x_train.reshape(-1,224,224,3)
        y_train = np_utils.to_categorical(np.array(y_train),num_classes=2)
        yield (x_train,y_train)

if __name__=="__main__":
    log_dir = "./logs/"

    with open(r".\data\dataset.txt","r") as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines) #打乱
    np.random.seed(None)

    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    model = AlexNet()

    #每3次保存一次模型
    checkpoint_period1 = ModelCheckpoint(
        log_dir+'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc', #准确率
        save_weights_only=False,
        save_best_only=True, # 只保存表现最好的模型
        period=3
    )

 #当准确率在3个epoch内不提升时，减小学习率
    reduce_lr = ReduceLROnPlateau(monitor='acc',factor=0.5,patience=3,verbose=1)

    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    # 在验证损失（val_loss）在10个epoch内没有下降时，提前停止训练。

    early_stopping = EarlyStopping(
        monitor='val_loss',# 监控的指标为验证损失
        min_delta=0, # 验证损失的变化量小于等于0时不算作改善
        patience=10,
        verbose=1)# 触发早停时 会打印相关信息

   #编译模型
    model.compile(
        loss='categorical_crossentropy',  # 交叉熵
        optimizer=Adam(lr=1e-3), # 优化器
        metrics=['accuracy']) #准确率

    # 一次的训练集大小
    batch_size = 128

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    #fit_generator方法适用于数据量较大，不能一次性加载到内存中的情况，通过生成器逐步生成训练数据。
    model.fit_generator(generate_arrays(lines[:num_train], batch_size), #生成训练数据
                        steps_per_epoch=max(1, num_train // batch_size), #每个epoch的步骤数
                        validation_data=generate_arrays(lines[num_train:], batch_size), # 生成验证数据
                        validation_steps=max(1, num_val // batch_size), # 每个验证epoch的步骤数
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr]) #回调函数
    model.save_weights(log_dir + 'last1.h5')

