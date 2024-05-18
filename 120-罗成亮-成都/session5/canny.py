import numpy as np
from matplotlib import pyplot as plt


def gray(img):
    return img.mean(axis=-1)


def gaussian_filter(img, dx, dy):
    filter = [
        [1 / 273, 4 / 273, 7 / 273, 4 / 273, 1 / 273],
        [4 / 273, 16 / 273, 26 / 273, 16 / 273, 4 / 273],
        [7 / 273, 26 / 273, 41 / 273, 26 / 273, 7 / 273],
        [4 / 273, 16 / 273, 26 / 273, 16 / 273, 4 / 273],
        [1 / 273, 4 / 273, 7 / 273, 4 / 273, 1 / 273],
    ]
    img_new = np.zeros(img.shape)
    tmp = 5 // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + 5, j:j + 5] * filter)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.axis('off')
    return img_new


def sobel(img, dx, dy):
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img.shape)
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img.shape)
    img_pad = np.pad(img, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y / img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')
    return [img_tidu, angle]


# Non-Maximum Suppression
def nms(img, angle, dx, dy):
    img_new = np.zeros(img.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True
            temp = img[i - 1:i + 2, j - 1:j + 2]
            if angle[i, j] <= -1:
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img[i, j] > num_1 and img[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img[i, j] > num_1 and img[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img[i, j] > num_1 and img[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img[i, j] > num_1 and img[i, j] > num_2):
                    flag = False
            if flag:
                img_new[i, j] = img[i, j]
    plt.figure(3)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.axis('off')
    return img_new


def connect_edges(img):
    lower_boundary = img.mean()
    high_boundary = lower_boundary * 3
    stack = []
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j] >= high_boundary:
                img[i, j] = 255
                stack.append([i, j])
            elif img[i, j] <= lower_boundary:
                img[i, j] = 0

    while not len(stack) == 0:
        temp_1, temp_2 = stack.pop()
        a = img[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img[temp_1 - 1, temp_2 - 1] = 255
            stack.append([temp_1 - 1, temp_2 - 1])
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img[temp_1 - 1, temp_2] = 255
            stack.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img[temp_1 - 1, temp_2 + 1] = 255
            stack.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img[temp_1, temp_2 - 1] = 255
            stack.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img[temp_1, temp_2 + 1] = 255
            stack.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img[temp_1 + 1, temp_2 - 1] = 255
            stack.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img[temp_1 + 1, temp_2] = 255
            stack.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img[temp_1 + 1, temp_2 + 1] = 255
            stack.append([temp_1 + 1, temp_2 + 1])

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] != 0 and img[i, j] != 255:
                img[i, j] = 0
    plt.figure(4)
    plt.imshow(img.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()
    return img


if __name__ == '__main__':
    pic_path = '../lenna.png'
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.png':
        img = img * 255
    img_new = gray(img)
    dx, dy = img_new.shape
    img_new = gaussian_filter(img_new, dx, dy)
    [img_new, angle] = sobel(img_new, dx, dy)
    img_new = nms(img_new, angle, dx, dy)
    connect_edges(img_new)
