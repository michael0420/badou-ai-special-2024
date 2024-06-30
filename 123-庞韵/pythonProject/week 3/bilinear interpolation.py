import numpy as np
import cv2


# python implementation of bilinear interpolation
# Define the function.
def bilinear_interpolation(img, out_dim):
    # original size, also need to get the channel
    # the image representation in numpy array is img.shape(height, width, channel)
    src_h, src_w, channel = img.shape
    # target size
    # diff than img.shape, during image processing, the image data is given as (width, height)
    dst_h, dst_w = out_dim[1], out_dim[0]
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)
    # first check if the original size is the same as target size
    # then return a copy, not original to avoid potential bug
    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    # create an array filled with zeros with defined type
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

    # get the scale factor
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h

    # loop over the three channels
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                # to avoid image shifting, and to align the new image with the original image,
                # need to adjust the coordinate system via geometric center symmetry
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # Find the coordinates of the surrounding points
                # the four points are (src_x0, src_y0)  (src_x1, src_y0) (src_x0, src_y1)  (src_x1, src_y1)
                src_x0 = int(np.floor(src_x))  # np.floor makes sure it is within the boundary and
                                                # 向下取整, 返回不大于输入参数的最大整数
                src_y0 = int(np.floor(src_y))
                src_x1 = min(src_x0 + 1, src_w-1)
                src_y1 = min(src_y0 + 1, src_h-1)


                # find the interpolated values temp0 and temp1
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0,src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                # find the target value from temp0 and temp1
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y)*temp0 + (src_y - src_y0)*temp1)
    return dst_img


# implement the function
if __name__ == '__main__':
    img = cv2.imread('../images/lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('Bilinear interpolation', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
