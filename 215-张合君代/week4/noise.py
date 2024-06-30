import random

import cv2 as cv
import numpy as np
from skimage import util


def salt_n_pepper_noise(image, perc):
    """
    Add salt&pepper noise with percentage. salt and pepper ration is 1:1.

    :param image: original/source image
    :param perc: float, the percentage of salt&pepper noise (pixel=255)
    :return: the image after adding noise
    """

    noise_count = int(image.shape[0] * image.shape[1] * float(perc))
    if noise_count == 0:
        return image

    (b, g, r) = cv.split(image);
    for i in range(noise_count):
        coor_x = random.randint(0, image.shape[0] - 1)
        coor_y = random.randint(0, image.shape[1] - 1)
        is_salt = random.random() > 0.5
        b[coor_x, coor_y] = 255 if is_salt else 0
        g[coor_x, coor_y] = 255 if is_salt else 0
        r[coor_x, coor_y] = 255 if is_salt else 0

    return cv.merge((b, g, r))


def gaussian_noise(image, means, sigma, perc):
    """
    Add gaussian noise with percentage.

    :param image: original/source image
    :param means: means to generate random gauss shade
    :param sigma: sigma to generate random gauss shade
    :param perc: the percentage of noise
    :return: the image after adding noise
    """
    noise_count = int(image.shape[0] * image.shape[1] * float(perc))
    if noise_count == 0:
        return image

    (b, g, r) = cv.split(image);
    for i in range(noise_count):
        coor_x = random.randint(0, image.shape[0] - 1)
        coor_y = random.randint(0, image.shape[1] - 1)
        factor = random.gauss(means, sigma)

        b[coor_x, coor_y] = np.clip(b[coor_x, coor_y] + factor, 0, 255)
        g[coor_x, coor_y] = np.clip(g[coor_x, coor_y] + factor, 0, 255)
        r[coor_x, coor_y] = np.clip(r[coor_x, coor_y] + factor, 0, 255)

    return cv.merge((b, g, r))


if __name__ == "__main__":
    before = cv.imread("alex.jpg");
    cv.imwrite('result/salt_n_pepper_noise.jpg', salt_n_pepper_noise(before, 0.3))
    cv.imwrite('result/gaussian_noise.jpg', gaussian_noise(before, 2, 10, 0.8))
    # Map the values to the range [0, 255] and Convert to uint8
    cv.imwrite('result/util_poisson.jpg', (util.random_noise(before, mode='poisson') * 255).astype(np.uint8))
    cv.imwrite('result/util_s&p.jpg',
               (util.random_noise(before, mode='s&p', amount=0.3) * 255).astype(np.uint8))
    cv.imwrite('result/util_speckle.jpg',
               (util.random_noise(before, mode='speckle') * 255).astype(np.uint8))
    cv.waitKey(0)
