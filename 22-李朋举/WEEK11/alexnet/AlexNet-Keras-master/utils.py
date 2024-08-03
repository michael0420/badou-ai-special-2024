import matplotlib.image as mpimg
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.ops import array_ops

'''
加载图像并进行裁剪。
1. `mpimg.imread(path)`：使用 `mpimg` 库的 `imread` 函数读取图像文件。`path` 是图像文件的路径。
2. `short_edge = min(img.shape[:2])`：获取图像的短边长度。`img.shape[:2]` 表示图像的前两个维度（通常是高度和宽度），`min` 函数返回这两个维度中的最小值。
3. `yy = int((img.shape[0] - short_edge) / 2)` 和 `xx = int((img.shape[1] - short_edge) / 2)`：
         计算裁剪图像的左上角坐标。通过将图像的高度和宽度分别减去短边长度，然后除以 2，得到裁剪区域的中心坐标。
4. `crop_img = img[yy: yy + short_edge, xx: xx + short_edge]`：使用切片操作裁剪图像。`img[yy: yy + short_edge, xx: xx + short_edge]` 表示从图像的 `(yy, xx)` 坐标开始，
         截取大小为 `short_edge` 的正方形区域。
5. 最后，函数返回裁剪后的图像。
总结起来，这个函数的作用是读取图像文件，并将其裁剪为中心的正方形。裁剪后的图像可以用于后续的图像处理或分析任务。
'''
def load_image(path):
    # 读取图片，rgb
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img

'''
调整图像的大小:
1. `with tf.name_scope('resize_image'):`：这是 TensorFlow 中的一个上下文管理器，用于给代码块添加一个名称范围。在这里，名称范围被设置为 `resize_image`。
2. `images = []`：创建一个空列表 `images`，用于存储调整大小后的图像。
3. `for i in image:`：遍历输入的图像列表 `image`。
4. `i = cv2.resize(i, size)`：使用 OpenCV 的 `resize` 函数对当前图像进行调整大小。`size` 是一个元组，表示调整后的图像大小。
5. `images.append(i)`：将调整大小后的图像添加到 `images` 列表中。
6. `images = np.array(images)`：将 `images` 列表转换为 NumPy 数组。
7. `return images`：函数返回调整大小后的图像数组。
总结起来，这个函数的作用是对输入的图像列表进行逐个调整大小，并将结果存储在一个 NumPy 数组中返回。
'''
def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)  # i(224，224，3) -> size(224,224)
            images.append(i)
        images = np.array(images)
        return images
'''
用于打印答案:
1. `with open("./data/model/index_word.txt","r",encoding='utf-8') as f:`：打开一个文件，并将文件对象赋值给变量 `f`。文件的路径为 `"./data/model/index_word.txt"`，打开模式为 `r`（只读），编码为 `utf-8`。
2. `synset = [l.split(";")[1][:-1] for l in f.readlines()]`：使用列表推导式从文件中读取每一行，并将每一行的第二个元素（通过 `split(";")` 分割）的最后一个字符去掉，然后将结果存储在列表 `synset` 中。
3. `print(synset[argmax])`：打印列表 `synset` 中索引为 `argmax` 的元素。
4. `return synset[argmax]`：返回列表 `synset` 中索引为 `argmax` 的元素。
总结起来，这个函数的作用是从指定文件中读取数据，并根据输入的索引 `argmax` 打印和返回相应的元素。
'''
def print_answer(argmax):
    with open("./data/model/index_word.txt","r",encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]
        
    print(synset[argmax])
    return synset[argmax]