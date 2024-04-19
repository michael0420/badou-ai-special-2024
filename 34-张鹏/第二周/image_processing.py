import cv2

def nearest_neighbor_interpolation(image, scale):
    height, width = image.shape[:2]
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return new_image

if __name__ == "__main__":
    # 读取图片
    image = cv2.imread('164533-16953723335065.jpg')

    # 灰度化
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # 最临近插值
    scaled_binary_image = nearest_neighbor_interpolation(image, 2.0)

    # 显示原始图片、灰度化图片、二值化图片和插值后的图片
    cv2.imshow('Original Image', image)
    cv2.imshow('Gray Image', gray_image)
    cv2.imshow('Binary Image', gray_image)
    cv2.imshow('Scaled Binary Image', scaled_binary_image)

    # 保存处理后的图片
    cv2.imwrite('gray_image.jpg', gray_image)
    cv2.imwrite('gray_image.jpg', gray_image)
    cv2.imwrite('scaled_binary_image.jpg', scaled_binary_image)
