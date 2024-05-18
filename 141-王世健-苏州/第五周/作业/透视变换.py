# Import libraries
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
# Load the image
image = imread('distortion.png')
# Source points
src = np.array([169, 115,                    # top left
                81, 464,                   # bottom left
                613, 8,                    # top right
                646, 522]).reshape((4, 2)) # bottom right

# Estimate the width and height from the source points
width = np.max(src[:, 0]) - np.min(src[:, 0])
height = np.max(src[:, 1]) - np.min(src[:, 1])
# Destination points (forming a box shape)
dst = np.array([
    [0, 0],
    [0, height],
    [width, 0],
    [width, height]
])
# Compute the projective transform
tform = transform.estimate_transform('projective', src, dst)
# Apply the transformation
warped_image = transform.warp(image, tform.inverse, output_shape=(height, width))
# Convert the warped image to uint8
warped_image_uint8 = (warped_image * 255).astype(np.uint8)
# Display the transformed and cropped image
plt.figure(figsize=(20,20))
plt.imshow(warped_image_uint8)
plt.title('Transformed and Cropped Image', fontsize=20, weight='bold')
plt.axis('off')
plt.show()