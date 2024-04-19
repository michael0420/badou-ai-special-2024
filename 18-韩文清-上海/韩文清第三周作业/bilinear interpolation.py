import numpy as np
import cv2

def bilinear_interpolation(img, out_dim):
    origin_h, origin_w, channel = img.shape
    target_h, target_w = out_dim[1], out_dim[0]

    if origin_h == target_h and origin_w == target_w:
        return img.copy()

    target_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # 计算原图和目标图的比例关系
    scale_x = float(origin_w) / target_w
    scale_y = float(origin_h) / target_h

    for y in range(target_h):
        for x in range(target_w):
            # 计算在原图中的坐标，中心对称
            origin_x = (x + 0.5) * scale_x - 0.5
            origin_y = (y + 0.5) * scale_y - 0.5
            '''采用目标图像中像素位置的中心点作为计算参考点，并使用尺寸比例关系来确定在原始图像中的坐标。'''
            # 找到在原图中相邻的四个像素点的坐标
            x0 = int(np.floor(origin_x))
            y0 = int(np.floor(origin_y))
            x1 = min(x0 + 1, origin_w - 1)
            y1 = min(y0 + 1, origin_h - 1)
            '''根据原始图像中的坐标，找到在原始图像中与目标图像中像素位置最接近的四个像素点的坐标。
            由于双线性插值需要在原始图像中找到四个相邻的像素点，所以我们将使用floor函数向下取整来确定这四个像素点的位置，
            并使用min函数确保这些像素点的位置不超出原始图像的边界。'''
            # 计算插值权重
            dx = origin_x - x0
            dy = origin_y - y0
            '''在这一步，我们计算出目标像素位置在原始像素格中的偏移量，即在原始像素格中的位置与整数坐标之间的距离。
            假设目标像素在(x0, y0)处的偏移量分别为dx和dy。那么目标像素在(x1, y0)处的权重为(1-dx, dy)，
            在(x0, y1)处的权重为(dx, 1-dy)，在(x1, y1)处的权重为(dx, dy)。'''
            # 双线性插值
            for c in range(channel):
                target_img[y, x, c] = (1 - dx) * (1 - dy) * img[y0, x0, c] + dx * (1 - dy) * img[y0, x1, c] + \
                                       (1 - dx) * dy * img[y1, x0, c] + dx * dy * img[y1, x1, c]
            '''最后，我们使用双线性插值公式来计算目标图像中像素的值。
            对于每个通道，我们计算目标像素位置与四个相邻像素位置之间的插值，并使用相应的插值权重对这些值进行加权求和。
            这样就得到了目标图像中当前像素的值。'''
    return target_img

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    target_img = bilinear_interpolation(img, (900, 900))
    cv2.imshow('Bilinear Interpolation', target_img)
    cv2.waitKey()
    cv2.destroyAllWindows() #关闭所有窗口
