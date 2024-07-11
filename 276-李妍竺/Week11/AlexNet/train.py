from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K
K.set_image_data_format('channels_last')
#K.set_image_dim_ordering('tf')    # 返回默认的图像的唯独顺序 tf:hwc,  th:chw

'''
1.从文件中获取图像数据与标签
'''
def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)  # 有多少行，也就是有几个数据
    print('n', n)
    print('batchsize', batch_size)
    i = 0
    while 1:
        X_train = []
        Y_train = []

        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]   #将每行文本以分号作为分隔符分割，并从中提取第一个元素，作为图像名称存在 name 变量中。

            #读取图像
            img = cv2.imread(r".\data\image\train" + '/' + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255  # 归一化
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])    #[1]代表标签

            # 读完一个周期后重新开始
            i = (i + 1) % n  # 开始下一个batch,不会再一次打乱顺序

        #调用 写好的utils 进行 图像处理
        X_train = utils.resize_image(X_train,(224,224))
        X_train = X_train.reshape(-1,224,224,3)  # -1代表的是 224 224 3 形状的图有几个
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes=2)  #one hot 两类[0,1]
        '''
        return：一次返回一个值，执行完该行代码后，函数就结束了。
        yield：返回一个值后，函数暂停执行，下一次调用时会从暂停的地方继续执行。
        '''
        yield (X_train, Y_train)


if __name__ == "__main__":
    # 模型保存
    log_dir = "./logs/"

    # 读取文件中的数据
    with open(".\data\dataset.txt","r") as f:
        lines = f.readlines()

    # 打乱行的顺序
    np.random.seed(10101)  #确保相同
    np.random.shuffle(lines)
    np.random.seed(None)   #重置为系统时间

    # 90%用于训练，10%用于估计。
    num_train = int(len(lines)*0.9)
    num_val = len(lines) - num_train

    # AlexNet模型
    model = AlexNet()

    #保存的方式，学习率下降的方式，是否需要早停，交叉熵
    '''
    保存训练模型：
    keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    
    filepath: log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'：
            log_dir：日志目录的路径。模型将保存在这个目录下。
            'ep{epoch:03d}'：文件名模板。 {epoch:03d} 表示将epoch号以3位数字格式嵌入到文件名中。例如，第 2 个epoch的模型将命名为 ep002.h5。
            'loss{loss:.3f}'：文件名模板。 {loss:.3f} 表示在文件名中嵌入当前的训练损失，小数点后保留 3 位数。
            'val_loss{val_loss:.3f}.h5'：文件名模板。 {val_loss:.3f} 表示在文件名中嵌入当前的验证集损失，小数点后保留 3 位数。
    monitor='acc'：指定要监测的指标。在这一情况下，将监测准确率（acc）
    save_weights_only=False：一个布尔值参数。如果设为 True，模型只保存权重，不保存整个模型配置。在这一情况中，设为 False，即保存整个模型。
    save_best_only=True：一个布尔值参数。如果设为 True，只保存监测指标最好的模型。如果设为 False，则每 period 个epoch都保存模型。
    period=3：指定多长时间（以epoch为单位）保存一次模型。在这一情况下，每 3 个epoch保存一次模型文件。
    
    学习率下降：
    reduceLr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=50,verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.0001)
    
    monitor：监测的值，可以是accuracy，val_loss,val_accuracy。
    factor：缩放学习率的值，学习率将以lr = lr*factor的形式被减少
    patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    verbose： 整数。0：安静，1：更新信息。
    mode：‘auto’，‘min’，‘max’之一 默认‘auto’就行.如果是min模式，如果被监测的数据已经停止下降，学习速率会被降低；在max模式，如果被监测的数据已经停止上升,学习速率会被降低；在auto模式，方向会被从被监测的数据中自动推断出来。
    epsilon：阈值，用来确定是否进入检测值的“平原区”
    cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
    min_lr：学习率最小值，能缩小到的下限
    '''
    # 保存的方式，4世代保存一次
    checkpoint_period = ModelCheckpoint(
        log_dir +'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=4
    )

    # 学习率下降的方式，acc4次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=4,
        verbose=1
    )

    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,  #最小变化
        patience=10,
        verbose=1
    )

    # 交叉熵  compile:编译模型
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=1e-3),
        metrics=['accuracy']
    )

    # 一批数据大小
    batch_size = 128
    print('Train on {} samples, val on {} samples, ,with batch size{}.'.format(num_train,num_val,batch_size))

    #开始训练
    model.fit_generator(
        generate_arrays_from_file(lines[:num_train],batch_size),
        steps_per_epoch=max(1,num_train//batch_size),  #每个epoch处理的最大批次数 //:整数除法，返回不大于结果的一个最大整数
        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
        validation_steps=max(1,num_val//batch_size),
        epochs=45,
        initial_epoch=0,
        callbacks=[checkpoint_period,reduce_lr]   #指定在训练过程中使用的回调函数
    )

    model.save_weights(log_dir+'last1.h5') #log_dir 目录下的 last1.h5 文件中
