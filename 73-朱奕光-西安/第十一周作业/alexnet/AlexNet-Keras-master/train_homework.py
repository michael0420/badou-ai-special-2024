import numpy as np
from keras import backend as K
import keras
from AlexNet_Keras_homework import AlexNet
from read_file import read_file

K.set_image_data_format('channels_last')

log_dir = './logs/'

with open('./data/dataset.txt', 'r') as file:
    data = file.readlines()

num_train = int(len(data)*0.9)
num_test = len(data) - num_train

model = AlexNet()

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=log_dir + '朱奕光model-ep{epoch:03d}-val_loss{val_loss:.3f}.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    period=1,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=3,
    verbose=1
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5,
    verbose=1
)

model.compile(loss='categorical_crossentropy',
              optimizer= keras.optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

"""
开始训练
"""
batch_size = 128
model.fit_generator(
    generator=read_file(data[:num_train], batch_size=batch_size),
    steps_per_epoch=max(1, num_train//batch_size),
    validation_data=read_file(data[num_train:], batch_size=batch_size),
    validation_steps=max(1, num_test//batch_size),
    callbacks=[checkpoint, reduce_lr],
    epochs=50,
    verbose=1,
    initial_epoch=0
)
model.save_weights(log_dir+'朱奕光last1.h5')