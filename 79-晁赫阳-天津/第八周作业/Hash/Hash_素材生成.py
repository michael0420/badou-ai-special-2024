import cv2 as cv
import numpy as np
from PIL import Image
import os.path as path
from PIL import ImageEnhance


def rotate(image):
    def rotate_bound(image, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv.warpAffine(image, M, (nW, nH))
    return rotate_bound(image, 45)


def enhance_color(image):
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    return enh_col.enhance(color)


def blur(image):
    return cv.blur(image, (15, 1))


def sharp(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    return cv.filter2D(image, -1, kernel=kernel)


def contrast(image):
    def contrast_brightness_image(src1, a, g):
        h, w, ch = src1.shape
        src2 = np.zeros([h, w, ch], src1.dtype)
        return cv.addWeighted(src1, a, src2, 1 - a, g)
    return contrast_brightness_image(image, 1.2, 1)


def resize(image):
    return cv.resize(image, (0, 0), fx=1.25, fy=1)


def light(image):
    return np.uint8(np.clip((1.3 * image + 10), 0, 255))


def save_img(image, img_name):
    cv.imwrite(img_name, image, [int(cv.IMWRITE_JPEG_QUALITY), 70])
    pass


def show_img(image):
    cv.imshow('image', image)
    cv.waitKey(0)
    pass


def main():
    data_img_name = '0216.jpg'

    img = cv.imread(data_img_name)

    if img is None:
        print(f"错误：无法加载图像 {data_img_name}")
        return

    img_light = light(img)
    img_resize = resize(img)
    img_contrast = contrast(img)
    img_sharp = sharp(img)
    img_blur = blur(img)
    img_color = enhance_color(Image.open(data_img_name))
    img_rotate = rotate(img)
    img_rotate1 = Image.open(data_img_name).rotate(45)

    save_img(img_light, "%s_light.jpg" % data_img_name.split(".")[0])
    save_img(img_resize, "%s_resize.jpg" % data_img_name.split(".")[0])
    save_img(img_contrast, "%s_contrast.jpg" % data_img_name.split(".")[0])
    save_img(img_sharp, "%s_sharp.jpg" % data_img_name.split(".")[0])
    save_img(img_blur, "%s_blur.jpg" % data_img_name.split(".")[0])
    img_color.save("%s_color.jpg" % data_img_name.split(".")[0])
    img_rotate1.save("%s_rotate.jpg" % data_img_name.split(".")[0])

    show_img(img_rotate)
    pass


if __name__ == '__main__':
    main()
