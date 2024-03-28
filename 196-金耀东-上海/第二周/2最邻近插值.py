import cv2
import numpy as np

# 图像矩阵归一化
def toNormalize(img, max_value=255.0):
    return img / max_value

# 图像矩阵反归一化
def inverseNormalize(normalized_img, max_value=255.0):
    return (normalized_img * max_value).astype(np.uint8)

# 最邻近插值算法
def nearestInterpolation(src_img, dst_height, dst_width):
    src_height, src_width, channels = src_img.shape
    dst_img = np.zeros([dst_height, dst_width, channels], dtype=float)

    # 图像尺寸坐标从0到size-1,因此计算图像缩放倍数时，分子和分母要减1
    sh, sw = (dst_height-1) / (src_height-1), (dst_width-1) / (src_width-1)
    for h in range(dst_height):
        for w in range(dst_width):
            for c in range(channels):
                dst_img[h, w, c] = src_img[ round(h/sh), round(w/sw), c ]
    return dst_img

# 缩放图像
def resizeImg(src_img, dst_height, dst_width):
    tmp_img = nearestInterpolation( toNormalize( src_img ), dst_height, dst_width)
    return inverseNormalize( tmp_img )

# 展示图像
def displayImg(winname, img):
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__" :
    img_path = "img/lenna.jpg"

    src_img = cv2.imread(img_path)
    dst_img_100x100 = resizeImg(src_img, 100, 100)
    dst_img_600x800 = resizeImg(src_img, 600, 800)

    displayImg("original 225x225", src_img)
    displayImg("resized 100x100", dst_img_100x100)
    displayImg("resized 600x800", dst_img_600x800)





