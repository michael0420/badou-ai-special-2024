# to grayscale a color picture #

# import packages #
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2


img = cv2.imread('../images/lenna.png')
h,w = img.shape[:2]    # only extract the first two elements, which are height and width

# use the np.zeros() function to create an array filled with zeros,
# the syntax is -> numpy.zeros(shape, dtype=float, order='C') #
img_gray = np.zeros([h, w], img.dtype)

for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)  # convert BGR to grayscale, note that 0.11 for B

# print(m)  # i，j is the position of pixel, and the three elements that m returns is the values of BRG of the pixel#
print(img_gray.shape)
print("image show gray: %s"%img_gray) # %s - space holder to insert the image values（array of matrix for color values）
cv2.imshow("image show gray from cv2", img_gray) # to display image in a window#

# until line 21, you cannot display image unless you add the following two lines of code.
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Above line 25,26 is used for open a single window to display an image.
# if you want to display multiple images at once, see below.
# Display the original image#
plt.subplot(221)  # define 2x2, and show image at position 1#
img = plt.imread('../images/lenna.png')
plt.imshow(img)
plt.title('Original Image', fontsize=12)  # define and display the name of plot#
# plt.show()  # must call plt.show() after imshow() to display plot#

# display the converted grayscale image
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale image', fontsize=12)
# plt.show()

# convert to grayscale with the build in function, instead of line 11-17
img_gray_two = rgb2gray(img)  # build-in function from scikit image library
img_gray_three = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # build-in function from openCV

plt.subplot(223)
plt.imshow(img_gray_two, cmap='gray')
plt.title('Grayscale from rgb2gray()')
plt.subplot(224)
plt.imshow(img_gray_three, cmap='gray')
plt.title('Grayscale from openCV function')

plt.tight_layout()
plt.savefig('grayscale.png')
plt.show()




