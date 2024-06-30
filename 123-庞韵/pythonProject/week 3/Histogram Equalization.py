# Histogram equalization is a technique in image processing used to improve the contrast of an image.
# It works by adjusting the intensity distribution of the image, making the histogram of the output image more uniform.
# step 1 - compute the histogram
# step 2 - compute the cumulative distribution function (CDF)
# step 3 - apply the transformation by using the CDF
# step 4 - output the equalized image


import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1- for one color channel
# read image and convert it to grayscale
img = cv2.imread("../images/lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# histogram equalization for grayscale
dst = cv2.equalizeHist(gray)

# histogram
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.figure()
plt.hist(dst.ravel(), 256)  # dst.ravel() flatten multi-Dimension to One Dimension. Do not create copy.
plt.savefig("Equalized Histogram.png")
plt.show()

# np.hstack() - horizontally stack arrays into one
# stack the original gray with the equalized one for easy observation
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
cv2.destroyAllWindows()


# 2 - histogram equalization for color picture
img2 = cv2.imread("../images/lenna.png", 1)  # 1 is default color mode, 0 is grayscale, -1 is read as-it
cv2.imshow("source image", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# split three channels to equalize the color image
(b, g, r) = cv2.split(img2)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

# combine three into one
result = cv2.merge((bH, gH, rH))

plt.figure()
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))  # another way to convert: plt.imshow(result[..., ::-1])
plt.savefig("Color image histogram equalization.png")
plt.show()

cv2.imshow("Color image histogram equalization", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

