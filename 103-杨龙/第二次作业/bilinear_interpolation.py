import numpy as np
import cv2

"""
implementation of bilinear interpolcation
"""

w_index = 0  # width index
h_index = 1  # height index
c_index = 2  # channel index
x_index = 0  # X-axis coordinate index
y_index = 1  # Y-axis coordinate index


def zoom_img(img, out_dim):
    if __is_shape_not_change(img, out_dim):
        return img.copy()

    dst_img = __init_img(img.shape[c_index], out_dim)
    scale_ratio = __calculate_scale_ratio(img, out_dim)
    __interpose_pixel(dst_img, img, scale_ratio)
    return dst_img


def __is_shape_not_change(img, out_dim):
    return  img.shape[w_index] == out_dim[w_index] and img.shape[h_index] == out_dim[h_index]


def __init_img(channel, out_dim):
    img_shape = __build_img_shape(channel, out_dim)
    return np.zeros(img_shape, dtype=np.uint8)


def __build_img_shape(channel, out_dim):
    return out_dim[w_index], out_dim[h_index], channel


def __calculate_scale_ratio(img, out_dim):
    return float(img.shape[w_index]) / out_dim[w_index], float(img.shape[h_index]) / out_dim[h_index]


def __interpose_pixel(img, src_img, scale_ratio):
    for channel_no in range(img.shape[c_index]):
        for dst_y_coord in range(img.shape[h_index]):
            for dst_x_coord in range(img.shape[w_index]):
                dst_coord = (dst_x_coord, dst_y_coord)
                img[dst_y_coord, dst_x_coord, channel_no] = __calculate_bilinear_interpolation_pixel(dst_coord, scale_ratio, src_img, channel_no)


def __calculate_bilinear_interpolation_pixel(dst_coord, scale_ratio, img, channel_no):
    src_coord = __calculate_source_coord(dst_coord, scale_ratio)
    src_coord0 = __calculate_source_coord0(src_coord)
    src_coord1 = __calculate_source_coord1(src_coord0, img.shape)

    return __calculate_billinear_pixel(
        src_coord, src_coord0, src_coord1, channel_no, img
    )


def __calculate_source_coord(dst_coord, scale_ratio):
    return (dst_coord[w_index] + 0.5) * scale_ratio[w_index] - 0.5, (dst_coord[h_index] + 0.5) * scale_ratio[h_index] - 0.5


def __calculate_source_coord0(src_coord):
    return int(np.floor(src_coord[x_index])), int(np.floor(src_coord[y_index]))


def __calculate_source_coord1(src_coord0, ori_img_shape):
    return min(src_coord0[x_index] + 1, ori_img_shape[w_index] - 1), min( src_coord0[y_index] + 1, ori_img_shape[h_index] - 1)


def __calculate_billinear_pixel(src_coord, src_coord0, src_coord1, channel_no, img):
    temp0 = (src_coord1[x_index] - src_coord[x_index]) * img[ src_coord0[y_index], src_coord0[x_index], channel_no]+ (src_coord[x_index] - src_coord0[x_index]) * img[src_coord0[y_index], src_coord1[x_index], channel_no]
    temp1 = (src_coord1[x_index] - src_coord[x_index]) * img[ src_coord1[y_index], src_coord0[x_index], channel_no] + (src_coord[x_index] - src_coord0[x_index]) * img[src_coord1[y_index], src_coord1[x_index], channel_no]
    return int((src_coord1[y_index] - src_coord[y_index]) * temp0+ (src_coord[y_index] - src_coord0[y_index]) * temp1)


if __name__ == "__main__":
    img = cv2.imread("lenna.png")
    dst_img = zoom_img(img, (700, 700))
    cv2.imshow("bilinear interp lenna", dst_img)
    cv2.waitKey()
