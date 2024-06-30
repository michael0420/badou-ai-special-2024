'''
Sobel edge detection is a widely used technique in image processing and computer vision to detect edges in an image.
It works by calculating the gradient of the image intensity at each pixel, which highlights regions of high spatial
frequency that correspond to edges.

The Sobel operator uses two 3x3 convolution kernels, one for detecting changes in the horizontal direction (x-axis)
and one for detecting changes in the vertical direction (y-axis).
These kernels are convolved with the image to produce two gradient images, which are then combined to find the edges.
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("../images/lenna.png", 0)  # read image in grayscale mode

# apply sobel function
# Since unit8 only in the range of [0,255],
# we use "cv2.CV_16S" data type to avoid overflow when dealing with negative gradient value from sobel function
# cv2.Sobel applies sobel operator to an image to calculate the gradient in specific direction
# the sobel operator uses convolution with a kernel to compute the gradient
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 1, 0 -> calculate the 1st derivative in the x direction
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 0, 1-> calculate the 1st derivative in the y direction

# convert the 16-bit back to 8-bit image data type which in a range of [0,255]
# cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
# src: Source array (the input image with gradients).
# dst (optional): Output array of the same size and type as src.
# alpha (optional): Scale factor.
# beta (optional): Added value.
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

# Use cv2.addWeighted() to combine the sobel in two directions
# dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
# src1: The first input array (image).
# alpha: Weight of the first array elements.
# src2: The second input array (image), which must be the same size and type as src1.
# beta: Weight of the second array elements.
# gamma: Scalar added to each sum.
# dst (optional): Output array that has the same size and type as src1.
# dtype (optional): Desired output array type.

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

plt.imsave("sobel edge detection.png", dst)



cv2.imshow("absX", absX)
cv2.imshow("absY", absY)
cv2.imshow("Result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
