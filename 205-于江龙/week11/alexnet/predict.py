import numpy as np
from keras import backend as K
from alex_net import AlexNet
from PIL import Image
import tensorflow as tf

def predict():
    model = AlexNet(input_shape=(224, 224, 3))
    model(tf.random.normal([1, 224, 224, 3]))
    model.load_weights('logs/trained_weights_final.weights.h5')
    img_path = 'data/test/test.jpg'
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    pred = model.predict(img)
    pred = np.argmax(pred, axis=1)
    print("cat" if pred[0] == 0 else "dog")
    print('Finished Prediction')

if __name__ == '__main__':
    predict()
