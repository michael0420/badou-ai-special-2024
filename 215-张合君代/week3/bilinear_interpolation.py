import cv2
import numpy as np


def bilinear_zoom(image, new_shape):
    height, width, channels = image.shape
    new_height, new_width = new_shape[0], new_shape[1]

    new_image = np.zeros((new_height, new_width, channels), np.uint8)

    for c in range(channels):
        for new_i in range(new_height):
            for new_j in range(new_width):
                i = (new_i + 0.5) * (float(height) / new_height) - 0.5
                j = (new_j + 0.5) * (float(width) / new_width) - 0.5

                i0 = int(np.floor(i))
                i1 = min(height - 1, int(np.ceil(i)))
                j0 = int(np.floor(j))
                j1 = min(width - 1, int(np.ceil(j)))

                # calculate the interpolation
                temp0 = (j1 - j) * image[i0, j0, c] + (j - j0) * image[i0, j1, c]
                temp1 = (j1 - j) * image[i1, j0, c] + (j - j0) * image[i1, j1, c]
                new_image[new_i, new_j, c] = int((i1 - i) * temp0 + (i - i0) * temp1)

    return new_image


def nearest_zoom_2d_factors(image, factors):
    """
    zoom the image
    :param factors: array of height and width factors to multiply the height/width of the original image
    :param image: original image
    :return: the image after zooming
    """
    height, width, channels = image.shape
    height_factor, width_factor = factors[0], factors[1]
    new_height, new_width = int(height * height_factor), int(width * width_factor)
    new_image = np.zeros((new_height, new_width, channels), np.uint8)

    # 使用数组切片进行像素遍历
    for i in range(new_height):
        for j in range(new_width):
            new_image[i, j] = image[int(i / height_factor), int(j / width_factor)]

    return new_image


def nearest_zoom_to_shape(image, new_shape):
    """
    :param image:
    :param new_shape:
    :return:
    """
    height, width = image.shape[:2]
    new_height, new_width = int(new_shape[0]), int(new_shape[1])
    return nearest_zoom_2d_factors(image, (new_height / height, new_width / width))


if __name__ == "__main__":
    before = cv2.imread('alex.jpg')
    nearest_zoom_to_shape = nearest_zoom_to_shape(before, (900, 900))
    bilinear_zoom = bilinear_zoom(before, (900, 900))
    # 保存缩放前后的图片
    cv2.imwrite('result/nearest_zoom_to_shape.jpg', nearest_zoom_to_shape)
    cv2.imwrite('result/bilinear_zoom.jpg', bilinear_zoom)

    # resized_image = cv2.resize(before, (800, 800), interpolation=cv2.INTER_LINEAR)
