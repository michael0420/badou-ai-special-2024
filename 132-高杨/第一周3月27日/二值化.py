import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('lenna.png')

print(img.shape)

h,w=img.shape[:2]
twovalueimg=np.zeros([h,w],img.dtype)



for i in range(h):
    for j in range(w):
        m=img[i,j] /255.

        for k in range(2):
            if m[k] >0.5:
                m[k]=255
            else:
                m[k]=0
        twovalueimg[i, j] = int(m[0] + m[1] + m[2])

     #  m= twovalueimg[i,j]   #灰度图转化
      # print(m.shape)
       #m=int(0.11*m[0],0.59*m[1],0.3*m[2])
print(m)
print(twovalueimg)


print("image show gray:  %s"%twovalueimg)

cv2.imshow('twovalue',twovalueimg)

plt.subplot(221)

img = plt.imread("lenna.png")
plt.imshow(img)


#显示图像
plt.show()

