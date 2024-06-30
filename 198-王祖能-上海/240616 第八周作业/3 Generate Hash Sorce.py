import cv2
import numpy as np
from PIL import Image  # python imaging library
from PIL import ImageEnhance
import os.path as path


def rotate(src):
    def rotate_bound(src, angle):
        h, w = src.shape[:2]
        cx, cy = int(w / 2), int(h / 2)
        M = cv2.getRotationMatrix2D([cx, cy], angle, 1.0)  # center/angle/scale分别为[旋转中心]， [逆时针旋转角度]，[缩放比例]得到坐标变换矩阵
        # M = [[cos ceita, sin ceita, (1-cos ceita)Xc+(-sin ceita)Yc],
        #      [-sin ceita, cos ceita, sin ceita Xc + (1-cos ceita)Yc]]
        print(M)
        cos = M[0, 0]
        sin = M[0, 1]
        print(cos, sin)
        print((1-cos)*cx - sin * cy)  # 验算M矩阵各参数

        w_new = int(w * cos + h * sin)  # 计算旋转后图像边界尺寸
        h_new = int(h * cos + w * sin)
        # x向左平移(负数向左,正数向右) 100 个像素
        # y向下平移(负数向上,正数向下) 100 个像素
        M[0, 2] = M[0, 2] + 1/2*(w_new - w)  # 原图中心（w/2, h/2), 旋转后放大了，（0， 0）点不变，应向右下平移图像中心至重合
        M[1, 2] = M[1, 2] + 1/2*(h_new - h)  # M[:,:n-1]表示线性变换，M[:,2]表示平移

        print(w_new, h_new)
        dst = cv2.warpAffine(src, M, dsize=[w_new, h_new])
        # warpAffine仿射变换函数，可实现旋转，平移，缩放；变换后的平行线依旧平行
        # warpPerspective透视变换函数，可保持直线不变形，但是平行线可能不再平行
        '''
        仿射变换， 对向量空间进行【线性变换+平移】，变换为另外空间
        warpAffine(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None)
        src: 输入图像 dst：输出图像
        M: 2×3的变换矩阵
        dsize: 变换后输出图像尺寸，如果为 Size() ，将与输入图像相同
        flag: 指定插值的方法，默认为线性插值。可用的选项有 INTER_NEAREST, INTER_LINEAR, INTER_CUBIC 等
        borderMode：边界像素外扩方式
        borderValue：边界像素插值，默认用0填充
        '''
        return dst

    return rotate_bound(src, 20)


def enhance_image(src):  # 色度增强
    dst = ImageEnhance.Color(src)
    color = 1.5
    enh_img = dst.enhance(color)
    # enh_img.show(title='enhance_image')
    return enh_img


def enhance_contrast_img(src):  # 对比度增强
    factor = 3
    dst = ImageEnhance.Contrast(src).enhance(factor)
    # dst.show()
    return dst


def enhance_sharpness_img(src):
    factor = 5
    dst = ImageEnhance.Sharpness(src).enhance(factor)
    # dst.show()
    return dst


def enhance_brightness_img(src):
    factor = 5
    dst = ImageEnhance.Brightness(src).enhance(factor)
    # dst.show()
    return dst


def blur(src):  # 均值模糊
    dst = cv2.blur(src, (1, 240))  # 卷积核w=240取均值，横向模糊竖向不变，有竖向拉伸的错觉
    return dst


def sharp(src):  # 图像锐化
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])  # 拉普拉斯算子做锐化，np.float32作用是？？？
    dst = cv2.filter2D(src, -1, kernel, delta=000)  # 自定义卷积核实现卷积操作，-1表示与原始图像使用相同的图像深度
    return dst


def resize(src):  # 图像尺寸调整
    dst = cv2.resize(src, (0, 0), None, fx=1.25, fy=1.0, interpolation=cv2.INTER_CUBIC)
    return dst


def light(src):  # 图像对比度、明暗调整
    # dst = np.clip((1.5 * src + 20), 0, 255)  # 截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值
    dst = (1.5 * src + 20)
    dst = np.array(dst, dtype=np.uint8)
    print(dst[0, 0, 0])
    return dst


def contrast(src):  # 图片和白图按权重调整合成，对比度、明暗调整
    def contrast_brightness_img(src1, a, g):  # a乘原图增大像素差值表现为对比度，g直接加上表现为亮度
        src2 = np.zeros(src.shape, np.uint8)  # 或者src1.dtype
        src2 = cv2.addWeighted(src1, a, src2, 1-a, gamma=g)  # dst = src1 * alpha + src2 * beta + gamma
        return src2
    dst = contrast_brightness_img(src, 1.5, 20)
    print(dst[0, 0, 0])
    return dst


def show(src):
    cv2.imshow('src', src)
    cv2.waitKey()
    cv2.destroyAllWindows()
    pass


def save(src, dst_name, folder_path):
    cv2.imwrite(path.join(folder_path, dst_name), src, [cv2.IMWRITE_JPEG_QUALITY, 100])
    # 保存的是 OpenCV 图像（多维数组），不是 cv2.imread() 读取的图像文件，保存格式由 filename 的扩展名决定的，与读取的图像文件的格式无关
    # 指定压缩质量，JPEG可以从0到100的质量（越高越好）。默认值为95
    # os.path.join(path, *path)方法用于将两个或者多个路径拼接到一起组成一个新的路径
    # path：表示要拼接的文件路径。如果各组件名首字母不包含’/’，则函数会自动加上
    # *paths：表示要拼接的多个文件路径，这些路径间使用逗号进行分隔。如果在要拼接的路径中，没有一个绝对路径，那么最后拼接出来的将是一个相对路径。
    # 返回值：拼接后的路径。
    pass


def main():
    img_name = 'lenna.png'
    folder_path = "./source"
    img_path = path.join(folder_path, img_name)
    print(img_path)

    img = cv2.imread(img_path)
    blur_img = blur(img)
    sharp_img = sharp(img)
    resize_img = resize(img)
    light_img = light(img)
    contrast_img = contrast(img)

    img_pil = Image.open(img_path, 'r')  # 用PIL库处理的图像也得用PIL读取
    # print(img_pil)  # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=512x512 at 0x381F198>
    enh_img = enhance_image(img_pil)
    enh_contr_img = enhance_contrast_img(img_pil)
    enh_sharp_img = enhance_sharpness_img(img_pil)
    enh_bright_img = enhance_brightness_img(img_pil)

    save(blur_img, "%s_blur.jpg" % img_name.split(".")[0], folder_path)
    save(sharp_img, "%s_sharp.jpg" % img_name.split(".")[0], folder_path)
    save(resize_img, "%s_resize.jpg" % img_name.split(".")[0], folder_path)
    save(light_img, "%s_light.jpg" % img_name.split(".")[0], folder_path)
    save(contrast_img, "%s_contrast.jpg" % img_name.split(".")[0], folder_path)  # 把图片名提取，提出后缀，附加到处理方式名字前

    rotate_img = rotate(img)
    save(rotate_img, '%s_rotate.jpg' % img_name.split('.')[0], folder_path)
    enh_img.save(path.join(folder_path, '%s_enh.jpg') % img_name.split('.')[0])
    enh_sharp_img.save(path.join(folder_path, '%s_enh_sharp.jpg' % img_name.split('.')[0]))  # %s的定位可以在path.join的内部或者外部均可
    # cv2.imshow('img_name', np.hstack([img, rotate_img]))
    # cv2.waitKey()


if __name__ == '__main__':
    main()
