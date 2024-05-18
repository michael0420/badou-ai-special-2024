import cv2
import random

def salt_pepper_noise(img, proportion):
    test_img = img.copy()
    h, w = img.shape[0:2]
    test_num = int(proportion * h * w)
    for i in range(test_num):
        randX = random.randint(0, h - 1)
        randY = random.randint(0, w - 1)

        if random.random() < 0.5:
            test_img[randX, randY] = 0  # 添加椒噪声
        else:
            test_img[randX, randY] = 255  # 添加盐噪声
    return test_img

img = cv2.imread("../lenna.png", 0)
end_img = salt_pepper_noise(img, 0.8)
cv2.imshow("Salt and Pepper Noise", end_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
