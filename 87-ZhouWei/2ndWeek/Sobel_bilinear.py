import cv2
import numpy as np
import matplotlib.pyplot as plt

def bilinear_interpolation(img, out_dim):
    src_h, src_w, channels = img.shape
    dst_h, dst_w = out_dim[0], out_dim[1]
    scale_h, scale_w = float(src_h)/dst_h, float(src_w)/dst_w
    newimg = np.zeros([dst_h, dst_w, channels], img.dtype)

    for dst_x in range(dst_w):
        for  dst_y in range(dst_h):
            src_x = (dst_x + 0.5)*scale_w - 0.5
            src_y = (dst_y + 0.5)*scale_h - 0.5
            src_x0 = min(int(np.floor(src_x)), src_w-1)
            src_x1 = min(src_x0+1, src_w-1)
            src_y0 = min(int(np.floor(src_y)), src_h-1)
            src_y1 = min(src_y0+1, src_h-1)
            for i in range(channels):
                temp0 = (src_x1-src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1-src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                newimg[dst_y, dst_x, i] = int((src_y1-src_y)*temp0 + (src_y - src_y0) * temp1)
    return newimg

image = cv2.imread('img.png')
row, col = image.shape[:2]

# -----------bilinear_interpolation ----------
newimage = bilinear_interpolation(image, (row*2, col*2))

# -----------sobel ----------
x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

fig = plt.figure(figsize=(15, 7))
fig.add_subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")

fig.add_subplot(2, 4, 2)
plt.imshow(cv2.cvtColor(absX, cv2.COLOR_BGR2RGB))
plt.title("absX")

fig.add_subplot(2, 4, 3)
plt.imshow(cv2.cvtColor(absY, cv2.COLOR_BGR2RGB))
plt.title("absY")

fig.add_subplot(2, 4, 4)
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.title("Result")

fig.add_subplot(2, 4, 5)
plt.imshow(cv2.cvtColor(newimage, cv2.COLOR_BGR2RGB))
plt.title("Bilinear_interpolation")

plt.show()