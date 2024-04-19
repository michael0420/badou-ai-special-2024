
import binarization
import matplotlib.pyplot as plt
import cv2

# test self implementation
img_colorless = binarization.convert_to_colorless_image("lenna.png")
cv2.imshow('Lenna colorless image', img_colorless)
# cv2.waitKey(0)

img_colorless_binary = binarization.convert_to_colorless_image_np("lenna.png")
plt.imshow(img_colorless_binary, cmap='gray')
plt.show()
