# 临近差值实现图片缩放
import cv2
import numpy as np


# define a resize function manually
def resize_function(img):
    height, width, channels = img.shape
    new_height, new_width = 800, 800
    empty_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    # 获取像素比例scaling factor
    # desired dimension / original dimension
    sh = new_height / height
    sw = new_width / width
    for i in range(new_height):
        for j in range(new_width):
            # get the corresponding resized-pixel coordinates in the original image based on the scaling factors
            # min() is used to ensure the value will not exceed the rang of original height and width
            # min(arg1, arg2), return the smallest value btw two arguments
            # arg1 is int(i/sh+0.5), arg2 is (height - 1)
            # x = min(int(i / sh + 0.5), height - 1)
            # y = min(int(j / sw + 0.5), width - 1)
            x = int(i / sh + 0.5)  # int()是向下取整，+0.5来手动四舍五入
            y = int(j / sw + 0.5)
            x = min(x, height - 1)
            y = min(y, width - 1)
            empty_img[i, j] = img[x, y]
    return empty_img


# import original image
ori_img = cv2.imread('../images/lenna.png')
zoom = resize_function(ori_img)

# resize with opencv build-in function
resized_img_cv2 = cv2.resize(ori_img, (800, 800), interpolation=cv2.INTER_NEAREST)

# Create windows - mac may not open the image window in proper size, use named.window function before imshow to adjust
cv2.namedWindow('Original Image', cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('Nearest Interpolation - Manual', cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow('Nearest Interpolation - OpenCV', cv2.WINDOW_GUI_NORMAL)

# Display the images
cv2.imshow("Original Image", ori_img)
cv2.imshow("Nearest Interpolation - Manual", zoom)
cv2.imshow('Nearest Interpolation - OpenCV', resized_img_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Original image size - ", ori_img.shape)
print("Nearest Interpolation size - ", zoom.shape)
print("Nearest Interpolation size - ", resized_img_cv2.shape)



