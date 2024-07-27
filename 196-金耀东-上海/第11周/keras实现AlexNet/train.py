from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from modules.global_params import PATH_TRIAN_DATASHEET, PATH_TEST_DATASHEET, IMG_SHAPE, PATH_LOGS_DIR
from modules import utils , dataset , models
import os

if __name__ == "__main__":
    # 创建模型
    model = models.AlexNet_BN()

    # 定义模型参数
    loss = categorical_crossentropy  # 损失函数: 交叉熵
    optimizer = Adam(0.001)  # 优化器
    metrics = ["accuracy"]  # 使用准确率作为评估指标
    num_epoch = 50  # 训练代数
    batch_size = 32  # 批次大小

    # 创建数据生成器:gen_train用于训练模型，gen_test用于评估模型
    img_paths_train , labels_train = dataset.get_imgpaths_labels(PATH_TRIAN_DATASHEET)
    img_paths_test, labels_test = dataset.get_imgpaths_labels(PATH_TEST_DATASHEET)
    gen_train = dataset.generator(img_paths_train, labels_train, batch_size, IMG_SHAPE)
    gen_test = dataset.generator(img_paths_test, labels_test, batch_size, IMG_SHAPE)

    # 设置保存方式：每3个epoch检查一次，保存准确率（acc)最高的模型
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(PATH_LOGS_DIR, "ep{epoch:03d}-loss{loss:.3f}-acc{accuracy:.3f}-val_loss{val_loss:.3f}-val_acc{val_accuracy:.3f}.h5"),
        monitor="accuracy",
        save_weights_only=True,
        save_best_only=True,
        period=3,
        verbose=1,  # 在callback执行时显示消息
    )

    # 设置学习率下降方式：连续3次准确率不在增大
    reduce_lr = ReduceLROnPlateau(
        mode="max", # "max"模式检测metric是否不再增大
        monitor='accuracy',
        factor=0.5, # 触发条件后lr*=factor
        patience=3, # 不再减小（或增大）的累计次数；
        verbose=1,
    )

    # 设置早停方式：连续10次val_loss不再降低，则停止训练
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1,
    )

    # 编译模型
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # 训练模型
    model.fit_generator(
        generator=gen_train,
        validation_data=gen_test,
        steps_per_epoch=max(1, len(labels_train)//batch_size),
        validation_steps=max(1, len(labels_train)//batch_size),
        epochs=num_epoch,
        callbacks=[checkpoint, reduce_lr, early_stopping],
    )
