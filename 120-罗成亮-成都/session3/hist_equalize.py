import cv2

if __name__ == '__main__':
    img = cv2.imread("../lenna.png")

    # 转换为 YCrCb 颜色空间
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # 分离通道
    y, cr, cb = cv2.split(img_yuv)

    # 只对 Y（亮度）通道进行直方图均衡化
    y_equalized = cv2.equalizeHist(y)

    # 将均衡化后的 Y 通道与原 Cr、Cb 通道合并
    img_yuv_equalized = cv2.merge([y_equalized, cr, cb])

    # 转回 BGR 颜色空间
    img_color_equalized = cv2.cvtColor(img_yuv_equalized, cv2.COLOR_YCrCb2BGR)

    # 显示或保存均衡化后的彩色图像
    cv2.imshow('COLOR_BGR2YCrCb', img_color_equalized)

    # 转换为 LAB 颜色空间（推荐使用 LAB 而非 YCrCb，因为 LAB 更符合人类视觉特性）
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 分离通道
    l, a, b = cv2.split(img_lab)

    # 创建 CLAHE 对象，设置参数（如 clipLimit 和 tileGridSize）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 只对 L（亮度）通道应用 CLAHE
    l_equalized = clahe.apply(l)

    # 将均衡化后的 L 通道与原 A、B 通道合并
    img_lab_equalized = cv2.merge([l_equalized, a, b])

    # 转回 BGR 颜色空间
    img_color_equalized = cv2.cvtColor(img_lab_equalized, cv2.COLOR_LAB2BGR)

    # 显示或保存均衡化后的彩色图像
    cv2.imshow('COLOR_BGR2LAB', img_color_equalized)

    cv2.waitKey()
    cv2.destroyAllWindows()
