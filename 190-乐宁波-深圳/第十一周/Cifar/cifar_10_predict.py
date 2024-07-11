import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt

# 加载模型
model = load_model('D:\\basic_cnn_model.h5')

# CIFAR-10 数据集类别映射
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 预处理图片
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32), interpolation="nearest")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# 进行预测并返回类别和概率
def predict_image(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    probabilities = predictions[0]
    return predicted_class, probabilities

# 假设你已经知道predict目录下有多少张图片
num_images = len(os.listdir('predict_data'))  # 举例：假设有5张图片，根据实际情况调整

# 遍历目录中的所有图片并收集结果
image_arrays = []
predicted_classes = []
for filename in os.listdir('predict_data'):
    if filename.endswith(".png"):  # 假设所有图片都是PNG格式
        file_path = os.path.join('predict_data', filename)
        img_array = preprocess_image(file_path)
        predicted_class, probabilities = predict_image(img_array)
        image_arrays.append(img_array)
        predicted_classes.append(predicted_class)

# 显示所有图片和预测结果
plt.figure(figsize=(num_images, 1))  # 根据图片数量调整图像窗口大小
for i, img_array in enumerate(image_arrays):
    plt.subplot(1, num_images, i + 1)  # 1行，图片数量列
    plt.imshow(img_array[0], cmap='gray')  # 显示图片，灰度图使用cmap='gray'
    plt.title(f'{class_names[predicted_classes[i]]}')  # 显示预测类别
    plt.axis('off')  # 不显示坐标轴

plt.tight_layout()  # 调整子图布局以适应图像和文字
plt.show()