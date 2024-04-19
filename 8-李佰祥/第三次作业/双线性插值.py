import cv2
import numpy
import numpy as np

img = cv2.imread("../../lenna.png")
src_h, src_w, c = img.shape
#目标大小
dest_h = 700
dest_w = 700

if src_h == dest_h and src_w == dest_w:
    img2 = img.copy()

dest_img = np.zeros((dest_h,dest_w,c),dtype=np.uint8)

scale_w, scale_h = float(src_w) / dest_w , float(src_h) / dest_h
for d_h in range(dest_h):
    for d_w in range(dest_w):
        #中心对称
        #根据目标图的坐标和缩放比例，计算该目标图的像素应该在原图的位置
        src_h = int((d_h+0.5) * scale_h -0.5)
        src_w = int((d_w+0.5) * scale_w -0.5)

        src_h0 = int(np.floor(src_h))
        src_h1 = min((src_h0 + 1),src_h-1)
        src_w0 = int(np.floor(src_w))
        src_w1 = min((src_w0 + 1), src_w - 1)

        temp0 = (src_w1 - src_w) * img[src_h0,src_w0] + (src_w - src_w0) * img[src_h0,src_w1]
        temp1 = (src_w1 - src_w) * img[src_h1,src_w0] + (src_w1 - src_w) * img[src_h1,src_w1]

        #print(temp0,temp1)
        dest_img[d_h,d_w] = (src_h1 - src_h)* temp0 + (src_h - src_h0)* temp1

cv2.imshow("img",img)
cv2.imshow("dest_img",dest_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


















