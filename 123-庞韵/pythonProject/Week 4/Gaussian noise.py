# Gaussian noise is a type of statistical noise having a probability density function (PDF)
# equal to that of the normal distribution, which is also known as the Gaussian distribution.
# 1.input sigma and mean
# 2.generate gaussian random number
# 3.generate output pixel from input
# 4. compress the pixel value back to the range of 0-255
# 5. loop through all pixel
# 6. output image


import cv2
import random
import matplotlib.pyplot as plt


def GaussianNoise(src, means, sigma, percentage):

    NoiseImage = src.copy()  # copy the source
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])  # get the total num of pixels

    for i in range(NoiseNum):
        # Generate a random point
        # use random.randint to randomly pick a number from total number of rows and cols ( -1 to avoid image edge)
        # the random pixel location comes from random x and y
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)

        # Add Gaussian noise (random.gauss) to the pixel value of the random point
        NoiseImage[randX, randY] = NoiseImage[randX, randY] + random.gauss(means,sigma)

        # Ensure the pixel values stay within the valid range [0, 255]
        if NoiseImage[randX,randY] < 0:
            NoiseImage[randX, randY] = 0
        elif NoiseImage[randX, randY] > 255:
            NoiseImage[randX, randY] = 255
    return NoiseImage


# Load the image in grayscale mode and apply the gaussian noise function
img = cv2.imread("../images/lenna.png", 0)
img1 = GaussianNoise(img,2,4,0.8)
plt.imsave("Gaussian_noise_grayscale.png", img1, cmap='gray')
plt.imshow(img1, cmap='gray')
plt.show()



