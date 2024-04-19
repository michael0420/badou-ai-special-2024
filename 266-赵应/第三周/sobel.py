import cv2
import tensorflow as tf

# sobel边缘检测


def corr2d_in_lesson(file_img):
    img = cv2.imread(file_img, cv2.IMREAD_GRAYSCALE)
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    abs_x = cv2.convertScaleAbs(x)
    abs_y = cv2.convertScaleAbs(y)
    dest = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    cv2.imshow('absX', abs_x)
    cv2.imshow('absY', abs_y)
    cv2.imshow('result', dest)
    cv2.waitKey(0)


# 单通道sobel边缘检测
def corr2d_single_channel(x, kernel):
    h, w = kernel.shape
    tmp = tf.Variable(tf.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1)))
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            tmp[i, j].assign(tf.reduce_sum(x[i: i + h, j: j + w] * kernel))
    return tmp


if __name__ == '__main__':
    corr2d_in_lesson('image/lenna.png')
    # gray_img = cv2.imread('image/lenna.png', cv2.IMREAD_GRAYSCALE)
    # tf.compat.v1.enable_eager_execution()
    # k = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
    # result = corr2d_single_channel(tf.convert_to_tensor(gray_img, dtype=tf.float32), k)
    # cv2.imshow("absX", result.numpy())
    # cv2.waitKey(0)
