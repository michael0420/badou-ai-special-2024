import cv2
import random
import numpy as np


def show_img(img, window_name, is_wait=False):
    cv2.imshow(window_name, img)
    if is_wait:
        cv2.waitKey(0)


def add_salt_pepper_noise(input_img, ratio=1):
    print("to add salt pepper noise ")
    img_noise = input_img.copy()
    ratio_num = int(ratio * input_img.shape[0] * input_img.shape[1])
    for i in range(0, ratio_num):
        rand_x = random.randint(0, input_img.shape[0] - 1)
        rand_y = random.randint(0, input_img.shape[1] - 1)
        tem = random.random()
        img_noise[rand_x, rand_y] = 255
        if tem > 0.5:
            img_noise[rand_x, rand_y] = 0
    return img_noise


def add_gaussian_noise(input_img, sigma, mean, ratio=1):
    print("to add gaussian noise ")
    img_noise = input_img.copy()
    ratio_num = int(ratio * input_img.shape[0] * input_img.shape[1])
    for i in range(0, ratio_num):
        rand_x = random.randint(0, input_img.shape[0] - 1)
        rand_y = random.randint(0, input_img.shape[1] - 1)
        img_noise[rand_x, rand_y] = img_noise[rand_x, rand_y] + random.gauss(sigma, mean)
        # if img_noise[rand_x, rand_y] > 255:
        #     img_noise[rand_x, rand_y] = 255
        # elif img_noise[rand_x, rand_y] < 0:
        #     img_noise[rand_x, rand_y] = 0
    img_noise = np.clip(img_noise, 0, 255)
    return img_noise


if __name__ == "__main__":
    src_img = cv2.imread('../data/lenna.png')
    src_img_gary = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    addNoiseImg1 = add_gaussian_noise(src_img_gary, 8, 5, 0.8)
    addNoiseImg2 = add_salt_pepper_noise(src_img_gary, 0.1)
    show_img(src_img_gary, "SrcImg")
    show_img(addNoiseImg1, "GaussianNoiseImg")
    show_img(addNoiseImg2, "SaltPepperNoiseImg", True)

