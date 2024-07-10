from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# 加载模型
model = load_model('D:\\basic_cnn_model.h5')

# CIFAR-10 数据集类别映射
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# 预处理图片
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# 进行预测并返回类别和概率
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    probabilities = predictions[0]

    # 打印每个类别的概率
    for i, class_name in enumerate(class_names):
        print(f'Probability of {class_name}: {probabilities[i]:.2f}')

    return predicted_class, probabilities


# 示例图片路径
img_path = 'img.png'
predicted_class, probabilities = predict_image(img_path)
print(f'The predicted class for the image is: {class_names[predicted_class]}')