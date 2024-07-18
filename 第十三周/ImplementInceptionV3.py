import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np

# 加载预训练的 InceptionV3 模型，包括顶层（全连接层用于分类）
base_model = InceptionV3(weights='imagenet', include_top=True)

# 如果你只需要特征提取器，可以设置为 include_top=False
# base_model = InceptionV3(weights='imagenet', include_top=False)

# 加载图像
# img_path = 'path_to_your_image.jpg'
img_path = '0003.webp'
img = image.load_img(img_path, target_size=(299, 299))

# 将图像转换为数组
x = image.img_to_array(img)

# 增加一个维度，因为模型期望的输入形状是 (batch_size, height, width, channels)
x = np.expand_dims(x, axis=0)

# 预处理图像
x = preprocess_input(x)

# 使用模型进行预测
preds = base_model.predict(x)

# 解码预测结果
print('Predicted:', decode_predictions(preds, top=3)[0])