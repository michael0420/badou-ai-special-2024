import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image

from mobilenet.Mobilenet import create_mobilenet


def input_normal(img):
    x = img / 255.
    x -= .5
    x *= 2
    return x


if __name__ == '__main__':
    model = create_mobilenet()
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)

    img_path = 'elephant.jpg'
    test_img = image.load_img(img_path, target_size=(224, 224))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    test_img = input_normal(test_img)

    preds = model.predict(test_img)
    print(np.argmax(preds))
    # 只显示top1
    print('Predicted:', decode_predictions(preds, 1))
