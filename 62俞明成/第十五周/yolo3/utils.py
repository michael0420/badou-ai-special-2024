from PIL import Image
import numpy as np
import tensorflow as tf


def scale_img(img, size):
    img_w, img_h = img.size
    w, h = size
    # 得到保持原图像比例，缩放后的宽高
    new_w = int(img_w * min(w * 1.0 / img_w, h * 1.0 / img_h))
    new_h = int(img_h * min(w * 1.0 / img_w, h * 1.0 / img_h))
    resize_img = img.resize((new_w, new_h), Image.BICUBIC)

    # 创建一个size大小的背景
    boxed_img = Image.new('RGB', size, (128, 128, 128))
    # 使其居中
    # 将resize_img调整到boxed_img的((w - new_w) // 2, (h - new_h) // 2)位置
    boxed_img.paste(resize_img, ((w - new_w) // 2, (h - new_h) // 2))
    return boxed_img


def load_weights(var_list, weights_file):
    """
    Introduction
    ------------
        加载预训练好的darknet53权重文件
    Parameters
    ----------
        var_list: 赋值变量名
        weights_file: 权重文件
    Returns
    -------
        assign_ops: 赋值更新操作
    """
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'conv2d' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'batch_normalization' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'conv2d' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


