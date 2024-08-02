from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K


# K.set_image_dim_ordering('tf')

# 从文件中读取数据，并以批量的形式生成训练数据和标签，以便进行深度学习模型的训练。
# 入参： lines（文件中的行列表）lines{list:22500} 用于训练的数据,已打乱顺序    和 batch_size（批量大小）128。
def generate_arrays_from_file(lines,batch_size):  # lines{list:22500} 用于训练的数据 已打乱顺序
    n = len(lines) # 获取文件的总行数 n: 22500
    i = 0 # 初始化一个指针 i 为 0
    while 1: # 无限循环
        X_train = []  # 用于存储当前批量的训练数据
        Y_train = []  # 用于存储当前批量的训练标签
        # 获取一个batch_size大小的数据  对于批量中的每个样本（从 0 到 batch_size-1）
        for b in range(batch_size):
            if i==0:  # 如果指针 i 为 0，对行列表进行随机洗牌，以确保数据的随机性
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]  # 根据行列表中的索引 i，获取文件名，并从文件中读取图像                    dog.503.jpg (图片)
            # 从文件中读取图像  将图像转换为 RGB 格式，并进行归一化处理（除以 255）
            img = cv2.imread(r".\data\image\train" + '/' + name)  # ndarray=(285,379,3)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # ndarray=(285,379,3)  图片大小不一样(h,w)不一样
            img = img/255  # ndarray=(285,379,3)
            X_train.append(img)  # 将处理后的图像添加到 X_train 列表中 {list:128} 128张h,w不同的3通道彩色图片
            Y_train.append(lines[i].split(';')[1])  # 将行列表中索引 i 处的标签添加到 Y_train 列表中{list:128}    ['1\n']   （标签）
            # 读完一个周期后重新开始  递增指针 i，并使用取模运算确保指针不会超出行列表的范围
            i = (i+1) % n
        # 处理图像 对图像进行处理，将128张图片每张图片其大小调整为 (224, 224)，并将其形状重塑为适合模型输入的格式。
        X_train = utils.resize_image(X_train,(224,224))
        X_train = X_train.reshape(-1,224,224,3)                              # (128,224,224,3)
        # 将标签进行独热编码，以便与模型的输出进行比较
        '''
        np.array(Y_train)：将标签数组 Y_train 转换为 NumPy 数组;  num_classes=2：指定了类别数量为 2。这意味着标签的取值范围是 0 到 1。
        np_utils.to_categorical(np.array(Y_train), num_classes=2)：调用 np_utils.to_categorical 函数将转换后的 NumPy 数组进行独热编码。
        独热编码是一种将离散标签转换为二进制向量的表示方法。在这种表示中，每个类别都对应一个二进制位，只有对应的位为 1，其他位为 0。
        '''
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= 2)  # (128,2)
        # 使用 yield 关键字生成一个元组 (X_train, Y_train)，表示当前批量的训练数据和标签
        yield (X_train, Y_train)  # {tuple:2}  0->(128,224,224,3)  1->(128,2)


if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r".\data\dataset.txt","r") as f:
        lines = f.readlines()  # ['cat.0.jpg;0\n', 'cat.1.jpg;0\n', ... 'dog.12999.jpg;1\n']

    # 打乱行，这个txt主要用于帮助读取数据来训练,打乱的数据更有利于训练
    '''
    打乱列表顺序：使用np.random.shuffle函数打乱列表lines的顺序。在打乱之前，设置了随机数种子为10101，以确保每次运行结果的可重复性。
               打乱完成后，将随机数种子设置为None，以恢复默认的随机数生成状态。 通过打乱数据的顺序，可以增加数据的随机性和多样性，避免模型过度依赖数据的顺序或出现过拟合的情况
    '''
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)     # 2500  用于估计
    num_train = len(lines) - num_val  # 22500 用于训练

    # 建立AlexNet模型
    model = AlexNet()
    
    # 保存的方式，3世代保存一次
    '''
    定义了一个 `ModelCheckpoint` 对象，用于在模型训练过程中进行检查点的保存。        
        - `log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'`：这是保存检查点文件的路径和文件名格式。`log_dir` 是日志目录，
                     `ep{epoch:03d}` 表示当前训练轮数，`loss{loss:.3f}` 表示当前训练损失，`val_loss{val_loss:.3f}` 表示当前验证损失。
        - `monitor='acc'`：指定要监控的指标，这里是准确率（`acc`）。
        - `save_weights_only=False`：表示是否只保存模型的权重。如果设置为 `True`，则只保存模型的权重；如果设置为 `False`，则保存整个模型。
        - `save_best_only=True`：表示是否只保存最优的检查点。如果设置为 `True`，则只有当监控指标的值优于之前保存的最优值时，才会保存检查点；如果设置为  `False`，则每次训练都会保存检查点。
        - `period=3`：表示每经过多少个训练轮数保存一次检查点。
        
        综上所述，这段代码的作用是在模型训练过程中，每经过 3 个训练轮数，保存一次检查点文件，文件名为当前训练轮数、训练损失、验证损失的组合，并根据准确率指标保存最优的检查点。这些检查点可以用于模型的恢复、继续训练或评估。
    '''
    checkpoint_period1 = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='acc', 
                                    save_weights_only=False, 
                                    save_best_only=True, 
                                    period=3
                                )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    '''
    定义了一个 `ReduceLROnPlateau` 对象，用于在模型训练过程中根据监控指标的值自动调整学习率。        
        - `monitor='acc'`：指定要监控的指标，这里是准确率（`acc`）。
        - `factor=0.5`：表示学习率衰减的因子。当监控指标在一定的轮数（由 `patience` 参数指定）内没有改善时，学习率将乘以这个因子。
        - `patience=3`：表示在学习率衰减之前，监控指标没有改善的轮数。
        - `verbose=1`：表示是否在控制台输出相关信息。
        综上所述，这段代码的作用是在模型训练过程中，根据准确率指标的值自动调整学习率。如果准确率在一定的轮数内没有改善，学习率将乘以 0.5。这样可以在训练过程中动态地调整学习率，以提高模型的性能。
    '''
    reduce_lr = ReduceLROnPlateau(
                            monitor='acc', 
                            factor=0.5, 
                            patience=3, 
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    '''
    用于在模型训练过程中实现早停（Early Stopping）
        `EarlyStopping` 是一种在训练过程中监控验证集上的指标，并在指标不再改善时提前停止训练的技术。这样可以避免过度训练，节省计算资源，并提高模型的泛化能力。        
        - `monitor='val_loss'`：指定要监控的指标。在这里，我们选择监控验证集上的损失（`val_loss`）。
        - `min_delta=0`：表示当监控指标的改善小于 `min_delta` 时，不被认为是真正的改善。将其设置为 0 意味着只要验证损失不再下降，就会触发早停。
        - `patience=10`：指定在验证损失不再改善的情况下，继续训练的轮数。在这个例子中，如果验证损失在连续 10 轮训练中都没有改善，训练将被提前停止。
        - `verbose=1`：控制是否在控制台输出早停的相关信息。设置为 1 表示输出详细信息。
        通过使用 `EarlyStopping`，可以在模型训练过程中自动检测验证集上的性能不再提高的情况，并及时停止训练，避免浪费时间和资源在不必要的训练上。这有助于找到模型的最佳训练轮数，并提高模型的性能和泛化能力。
    '''
    early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            min_delta=0, 
                            patience=10, 
                            verbose=1
                        )

    # 交叉熵
    '''
    使用`compile`方法来配置模型的训练过程。
        - `loss = 'categorical_crossentropy'`：指定模型的损失函数。在多类别分类问题中，通常使用`categorical_crossentropy`作为损失函数。它衡量了模型预测的类别概率分布与真实类别分布之间的差异。
        - `optimizer = Adam(lr=1e-3)`：指定模型的优化器。在这里，我们选择使用`Adam`优化器，它是一种常用的优化算法。`lr=1e-3`表示学习率，控制着模型参数更新的步长。
        - `metrics = ['accuracy']`：指定模型在训练和评估过程中要计算的指标。在这里，我们选择计算准确率（`accuracy`）作为评估指标。
        通过配置这些参数，模型将使用`categorical_crossentropy`损失函数进行优化，并使用`Adam`优化器来更新模型参数。在训练过程中，模型将计算准确率作为评估指标，以便评估模型的性能。
    '''
    model.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(lr=1e-3),
            metrics = ['accuracy'])

    # 一次的训练集大小
    batch_size = 128
    # num_train=22500   num_val = 2500   batch_size=128
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    # 开始训练  存储训练的模型
    '''
    使用`fit_generator`方法来训练模型。        
        - `generate_arrays_from_file(lines[:num_train], batch_size)`：生成训练数据的批量。`lines[:num_train]`表示训练数据的行数，`batch_size`表示每个批量的大小。  22500
        - `steps_per_epoch=max(1, num_train//batch_size)`：
                指定每个训练轮数的步数。在这里，我们将其设置为训练数据的数量除以批量大小，并向上取整。这样可以确保每个训练轮数都能遍历完所有的训练数据。   22500 // 128 ≈ 175
        - `validation_data=generate_arrays_from_file(lines[num_train:], batch_size)`：生成验证数据的批量。`lines[num_train:]`表示验证数据的行数，`batch_size`表示每个批量的大小。 2500
        - `validation_steps=max(1, num_val//batch_size)`：指定每个验证轮数的步数。在这里，我们将其设置为验证数据的数量除以批量大小，并向上取整。这样可以确保每个验证轮数都能遍历完所有的验证数据。
        - `epochs=50`：指定训练的轮数。
        - `initial_epoch=0`：指定初始的训练轮数。
        - `callbacks=[checkpoint_period1, reduce_lr]`：指定训练过程中的两个回回调函数：
                                                    `checkpoint_period1`用于在每个训练轮数结束时保存模型的检查点； `reduce_lr`用于在验证损失不再改善时降低学习率。
        最后，代码使用`save_weights`方法来保存模型的权重。
        总的来说，这段代码的作用是使用生成器函数来生成训练数据和验证数据的批量，并使用`fit_generator`方法来训练模型。在训练过程中，模型将使用`categorical_crossentropy`损失函数进行优化，
                并使用`Adam`优化器来更新模型参数。在训练过程中，模型将计算准确率作为评估指标，以便评估模型的性能。
    '''
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[checkpoint_period1, reduce_lr])
    model.save_weights(log_dir+'last1.h5')

    '''
    在科学计算和数据分析中，大规模数据集的存储和管理是一个重要的问题。HDF5（Hierarchical Data Format version 5）是一种用于存储和组织大型数据集的文件格式。
    Python 的 h5py 库是一个用于与 HDF5 文件交互的接口，它结合了 HDF5 的强大功能和 Python 的易用性，使得处理大型数据集变得更加方便和高效  
    import h5py
    # 打开hdf5文件
    hdf5_file = h5py.File('./logs/last1.h5', 'r')
    # 查看文件结构
    print(hdf5_file.keys())
    # 读取数据集
    dataset = hdf5_file['batch_normalization_1']
    # <KeysViewHDF5 ['batch_normalization_1', 'batch_normalization_2', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'dense_1', 'dense_2', 'dense_3', 'dropout_1', 'dropout_2', 'flatten_1', 'max_pooling2d_1', 'max_pooling2d_2', 'max_pooling2d_3']>
    data = dataset[()]
    # 打印数据
    print(data)
    # 关闭hdf5文件
    hdf5_file.close()          
    '''



    '''
    Epoch 1/50

    1/175 [..............................] - ETA: 14:05 - loss: 0.9624 - accuracy: 0.4844
    2/175 [..............................] - ETA: 12:04 - loss: 10.5540 - accuracy: 0.4961
    3/175 [..............................] - ETA: 11:32 - loss: 7.3176 - accuracy: 0.5234 
    ......
    11/175 [>.............................] - ETA: 9:18 - loss: 2.5119 - accuracy: 0.5185
    12/175 [=>............................] - ETA: 9:10 - loss: 2.3605 - accuracy: 0.5202
    ......
    174/175 [============================>.] - ETA: 3s - loss: 0.7933 - accuracy: 0.5577
    175/175 [==============================] - 573s 3s/step - loss: 0.7925 - accuracy: 0.5579 - val_loss: 0.6982 - val_accuracy: 0.4988
    
    ......
    
    Epoch 25/50
    1/175 [..............................] - ETA: 11:30 - loss: 0.0280 - accuracy: 0.9922
    2/175 [..............................] - ETA: 11:04 - loss: 0.0558 - accuracy: 0.9688
    ......
    11/175 [>.............................] - ETA: 9:53 - loss: 0.0447 - accuracy: 0.9808
    12/175 [=>............................] - ETA: 9:45 - loss: 0.0419 - accuracy: 0.9824
    ......
    174/175 [============================>.] - ETA: 3s - loss: 0.0395 - accuracy: 0.9848
    175/175 [==============================] - 580s 3s/step - loss: 0.0396 - accuracy: 0.9848 - val_loss: 0.5995 - val_accuracy: 0.8388
   
    ......
   
    Epoch 50/50

    1/175 [..............................] - ETA: 11:46 - loss: 0.0442 - accuracy: 0.9766
    2/175 [..............................] - ETA: 11:47 - loss: 0.0259 - accuracy: 0.9844
    ......
    11/175 [>.............................] - ETA: 10:20 - loss: 0.0089 - accuracy: 0.9957
    12/175 [=>............................] - ETA: 10:09 - loss: 0.0082 - accuracy: 0.9961
    ......
    174/175 [============================>.] - ETA: 3s - loss: 0.0233 - accuracy: 0.9927
    175/175 [==============================] - 585s 3s/step - loss: 0.0233 - accuracy: 0.9926 - val_loss: 0.9270 - val_accuracy: 0.8207

    每个Epoch品骏耗时 580s  X 50 个Epoch ≈  29000s  ≈ 8 h 
    '''

