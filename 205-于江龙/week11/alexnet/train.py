# Epoch 50/50 - accuracy: 0.9884 - loss: 0.0269 - val_accuracy: 0.8884 - val_loss: 0.8582 - learning_rate: 0.0010
from tensorflow.keras import callbacks, backend
from tensorflow.keras.optimizers import Adam
from alex_net import AlexNet
from data_load import data_load
import numpy as np

def train_model():
    log_dir = 'logs/'
    batch_size = 128
    epochs = 50

    train_loader, test_loader = data_load(batch_size)
    num_train = len(train_loader)  
    num_test = len(test_loader)
    print(f'num_train: {num_train}, num_test: {num_test}')
    model = AlexNet()

    # save the model each 3 epochs
    model_checkpoint = callbacks.ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.keras',
        monitor='acc',
        save_best_only=True,
        save_weights_only=False,
    )

    # leanring rate decay
    lr_decay = callbacks.ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )

    # stop early
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    model.compile(
        optimizer=Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_loader,
        validation_data=test_loader,
        epochs=epochs,
        callbacks=[model_checkpoint, lr_decay]
    )
    model.save_weights(log_dir + 'trained_weights_final.weights.h5')

    print("train finished")

if __name__ == '__main__':
    train_model()

