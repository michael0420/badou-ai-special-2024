import cv2
import numpy as np
import math


def show_img(input_img, window_name, is_wait=False):
    cv2.imshow(window_name, input_img)
    if is_wait:
        cv2.waitKey(0)


def canny_detail_test(input_img, lower_boundary = None, high_boundary = None):
    print("to canny...... ")
    # 1.bgr to gray
    img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    show_img(img_gray, "img_gray")

    # 2.gaussian soft
    dx, dy = img_gray.shape
    gaussian_size = 5
    sigma = 0.5
    gaussian_kernel = np.zeros((gaussian_size, gaussian_size))
    for i in range(gaussian_size):
        for j in range(gaussian_size):
            gaussian_kernel[i][j] = (1 / (2 * math.pi * sigma ** 2)) * math.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    res_img_gaussian = np.zeros(img_gray.shape)
    tmp = gaussian_size // 2
    img_pad = np.pad(img_gray, ((tmp, tmp), (tmp, tmp)), 'constant')
    for i in range(dx):
        for j in range(dy):
            res_img_gaussian[i][j] = np.sum(img_pad[i:i + gaussian_size, j:j + gaussian_size] * gaussian_kernel)
    show_img(res_img_gaussian.astype(np.uint8), "res_img_gaussian")

    # 3. use sobel to edge detection
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    res_sobel = np.zeros(res_img_gaussian.shape)
    res_sobel_x = np.zeros(res_img_gaussian.shape)
    res_sobel_y = np.zeros(res_img_gaussian.shape)
    tmp = sobel_kernel_x.shape[0] // 2
    img_pad = np.pad(res_img_gaussian, ((tmp, tmp), (tmp, tmp)), 'constant')
    for i in range(dx):
        for j in range(dy):
            res_sobel_x[i][j] = np.sum(
                img_pad[i:i + sobel_kernel_x.shape[0], j:j + sobel_kernel_x.shape[1]] * sobel_kernel_x)
            res_sobel_y[i][j] = np.sum(
                img_pad[i:i + sobel_kernel_x.shape[0], j:j + sobel_kernel_x.shape[1]] * sobel_kernel_y)
            res_sobel[i][j] = np.sqrt(res_sobel_x[i, j] ** 2 + res_sobel_y[i, j] ** 2)
    res_sobel_x[res_sobel_x == 0] = 0.0000000000001

    show_img(res_sobel.astype(np.uint8), "res_sobel")

    # 4. use NMS to reduce edge detection result
    res_nms = np.zeros(res_sobel.shape)
    tan_tidu = res_sobel_y / res_sobel_x
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            is_max = True
            temp = res_sobel[i - 1:i + 2, j - 1:j + 2]
            if tan_tidu[i, j] <= -1:
                num_1 = (temp[0, 1] - temp[0, 0]) / tan_tidu[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / tan_tidu[i, j] + temp[2, 1]
                if not (res_sobel[i, j] > num_1 and res_sobel[i, j] > num_2):
                    is_max = False
            elif tan_tidu[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / tan_tidu[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / tan_tidu[i, j] + temp[2, 1]
                if not (res_sobel[i, j] > num_1 and res_sobel[i, j] > num_2):
                    is_max = False
            elif tan_tidu[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * tan_tidu[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * tan_tidu[i, j] + temp[1, 0]
                if not (res_sobel[i, j] > num_1 and res_sobel[i, j] > num_2):
                    is_max = False
            elif tan_tidu[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * tan_tidu[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * tan_tidu[i, j] + temp[1, 2]
                if not (res_sobel[i, j] > num_1 and res_sobel[i, j] > num_2):
                    is_max = False
            if is_max:
                res_nms[i, j] = res_sobel[i, j]
    show_img(res_nms.astype(np.uint8), "res_nms")

    # 5. use double_boundary to reduce edge detection result
    res_double_boundary = res_nms.copy()
    if lower_boundary is None:
        lower_boundary = res_sobel.mean() * 0.5
        high_boundary = lower_boundary * 3
    low_edge_list = []
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            if res_nms[i, j] >= high_boundary:
                res_double_boundary[i, j] = 255
            elif res_nms[i, j] <= lower_boundary:
                res_double_boundary[i, j] = 0
            else:
                low_edge_list.append([i, j])
    for item in low_edge_list:
        pix_8 = res_double_boundary[item[0]-1:item[0]+2, item[1]-1:item[1]+2]
        if np.max(pix_8) > high_boundary:
            res_double_boundary[item[0], item[1]] = 255

    for i in range(res_double_boundary.shape[0]):
        for j in range(res_double_boundary.shape[1]):
            if res_double_boundary[i, j] != 0 and res_double_boundary[i, j] != 255:
                res_double_boundary[i, j] = 0
    show_img(res_double_boundary.astype(np.uint8), "res_double_boundary", True)

    return res_double_boundary


if __name__ == "__main__":

    img = cv2.imread('../data/lenna.png')
    # res_canny = canny_detail_test(img)
    res_canny = canny_detail_test(img, 100, 180)


