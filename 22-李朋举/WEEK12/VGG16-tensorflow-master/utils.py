import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

'''
定义`load_image`函数的代码片段，用于从给定路径加载图像并进行裁剪。
    - `mpimg.imread(path)`：使用`matplotlib.image`模块的`imread`函数读取图像文件。
    - `short_edge = min(img.shape[:2])`：获取图像的短边长度。
    - `yy = int((img.shape[0] - short_edge) / 2)` 和 `xx = int((img.shape[1] - short_edge) / 2)`：计算裁剪区域的左上角坐标。
    - `crop_img = img[yy: yy + short_edge, xx: xx + short_edge]`：使用切片操作裁剪图像。
总的来说，该函数的作用是读取图像文件，找到其短边，然后从图像中心裁剪出一个正方形区域。
请注意，这段代码可能需要在运行环境中安装`matplotlib`库。此外，还需要确保提供的路径是有效的图像文件路径。
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
定义`resize_image`函数的代码片段，用于调整图像的大小。
    - `image`：要调整大小的图像张量。
    - `size`：目标大小，通常是一个元组表示新的高度和宽度。
    - `method`：调整大小的方法，默认为`tf.image.ResizeMethod.BILINEAR`，还可以选择其他方法如最近邻插值。
    - `align_corners`：是否对齐图像的角落，默认为`False`。
函数内部使用`tf.name_scope`创建了一个命名空间`resize_image`。然后，通过`tf.expand_dims`在图像张量的第 0 维添加一个维度，使其成为一个 4D 张量。
接下来，使用`tf.image.resize_images`函数根据指定的大小和方法调整图像的大小。最后，使用`tf.reshape`将调整大小后的图像张量恢复为原始的形状，
除了第 0 维的维度被设置为`-1`，表示自动计算该维度的大小。
总的来说，该函数的作用是将输入的图像张量调整为指定的大小，并返回调整后的图像张量。
请注意，这段代码需要在 TensorFlow 环境中运行，并且可能需要根据实际情况进行适当的修改和调整。
'''
def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size,
                                       method, align_corners)
        image = tf.reshape(image, tf.stack([-1,size[0], size[1], 3]))
        return image

'''
定义`print_prob`函数的代码片段，用于打印图像的预测概率。
    - `prob`：这是一个一维数组，表示每个类别的预测概率。
    - `file_path`：这是一个文件路径，文件中包含了每个类别的名称。
    - `synset`：通过读取文件，将每个类别的名称存储在`synset`列表中。
    - `pred`：这是一个索引数组，表示预测概率从大到小排列的结果的序号。
    - `top1`：取预测概率最大的类别名称和概率。
    - `top5`：取预测概率最大的前 5 个类别名称和概率，并将它们存储在一个列表中。
    - `print(("Top1: ", top1, prob[pred[0]]))`：打印预测概率最大的类别名称和概率。
    - `print(("Top5: ", top5))`：打印预测概率最大的前 5 个类别名称和概率。
    - `return top1`：返回预测概率最大的类别名称。
总的来说，该函数的作用是读取类别名称文件，根据预测概率对类别进行排序，并打印预测概率最大的类别名称和概率，以及预测概率最大的前 5 个类别名称和概率。
'''
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    # 将概率从大到小排列的结果的序号存入pred
    pred = np.argsort(prob)[::-1]
    # 取最大的1个、5个。
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1



