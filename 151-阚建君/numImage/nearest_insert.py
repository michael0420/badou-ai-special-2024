import cv2
import numpy as np
def insert_up(img,high,wide):
    height, width, channels = img.shape
    changeImage = np.zeros((high, wide, channels), np.uint8)
    change_height = height/high    #缩放倍数
    change_width = width/wide
    for i in range(high):
        for j in range(wide):
            x = int(i*change_height + 0.5)  # 先向上加到整数，再向下取整找到最邻近的点 其实就是四舍五入
            y = int(j*change_width + 0.5)
            changeImage[i, j] = img[x, y]
    return changeImage

if __name__ == '__main__':
    # 输入图片比例
    high_str, wide_str = input("请输入高和宽，以空格分隔: ").split()
    high = int(high_str)
    wide = int(wide_str)
    img = cv2.imread("lenna.png")
    result = insert_up(img, high, wide)
    print(result)
    print(result.shape)
    cv2.imshow("after change", result)
    cv2.imshow("original", img)
    cv2.waitKey(0)