import cv2
import numpy as np

def n_interp(img):
    height,width, channels = img.shape[:3]
    emptyImg = np.zeros((1440,2560,channels),np.uint8)
    expMultW = 2560/width
    expMultH = 1440/height
    print(width,height,expMultW,expMultH)
    for j in range(1440):
        for i in range(2560):
            y = int(j / expMultH)
            x = int(i / expMultW)
            emptyImg[j,i] =img[y,x]
    return emptyImg


# cv2.resize(img, (2560,1440,c),near/bin)
img = cv2.imread("shangri-la.jpg")
print(img.shape)
expImg = n_interp(img)
print(expImg)
cv2.imshow("nearest interp",expImg)
cv2.imshow("source image",img)
cv2.waitKey(0)





