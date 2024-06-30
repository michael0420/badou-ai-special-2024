import cv2 as cv
import numpy as np
from PIL import Image
import os.path as path
from PIL import ImageEnhance


def rotate(image):
    def rotate_bound(image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv.warpAffine(image, M, (nW, nH))

    return rotate_bound(image, 45)


def enhance_color(image):
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    return enh_col.enhance(color)


def blur(image):
    # 模糊操作
    return cv.blur(image, (15, 1))


def sharp(image):
    # 锐化操作
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    return cv.filter2D(image, -1, kernel=kernel)


def contrast(image):
    def contrast_brightness_image(src1, a, g):
        """
        粗略的调节对比度和亮度
        :param src1: 图片
        :param a: 对比度
        :param g: 亮度
        :return:
        """

        # 获取shape的数值，height和width、通道
        h, w, ch = src1.shape

        # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
        src2 = np.zeros([h, w, ch], src1.dtype)
        # addWeighted函数说明如下
        return cv.addWeighted(src1, a, src2, 1 - a, g)

    return contrast_brightness_image(image, 1.2, 1)


def resize(image):
    # 缩放图片
    return cv.resize(image, (0, 0), fx=1.25, fy=1)


def light(image):
    # 修改图片的亮度
    return np.uint8(np.clip((1.3 * image + 10), 0, 255))


def save_img(image, img_name, output_path=None):
    # 保存图片
    cv.imwrite(path.join(output_path, img_name), image, [int(cv.IMWRITE_JPEG_QUALITY), 70])
    pass


def show_img(image):
    cv.imshow('image', image)
    cv.waitKey(0)
    pass


def main():
    data_img_name = 'lenna.png'
    output_path = "./source"
    data_path = path.join(output_path, data_img_name)

    img = cv.imread(data_path)

    # 修改图片的亮度
    img_light = light(img)
    # 修改图片的大小
    img_resize = resize(img)
    # 修改图片的对比度
    img_contrast = contrast(img)
    # 锐化
    img_sharp = sharp(img)
    # 模糊
    img_blur = blur(img)
    # 色度增强
    img_color = enhance_color(Image.open(data_path))
    # 旋转
    img_rotate = rotate(img)
    img_rotate1 = Image.open(data_path).rotate(45)
    # 两张图片横向合并（便于对比显示）
    # tmp = np.hstack((img, img_rotate))

    save_img(img_light, "%s_light.jpg" % data_img_name.split(".")[0], output_path)
    save_img(img_resize, "%s_resize.jpg" % data_img_name.split(".")[0], output_path)
    save_img(img_contrast, "%s_contrast.jpg" % data_img_name.split(".")[0], output_path)
    save_img(img_sharp, "%s_sharp.jpg" % data_img_name.split(".")[0], output_path)
    save_img(img_blur, "%s_blur.jpg" % data_img_name.split(".")[0], output_path)
    # save_img(img_rotate, "%s_rotate.jpg" % data_img_name.split(".")[0], output_path)
    # 色度增强
    img_color.save(path.join(output_path, "%s_color.jpg" % data_img_name.split(".")[0]))
    img_rotate1.save(path.join(output_path, "%s_rotate.jpg" % data_img_name.split(".")[0]))

    show_img(img_rotate)
    pass


if __name__ == '__main__':
    main()
