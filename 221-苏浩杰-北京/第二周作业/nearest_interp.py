import  cv2
import  numpy as np

def nearest(img):
    h,w,channel = img.shape
    newImage=np.zeros((300,300,channel),img.dtype)
    x = 300 / h
    y = 300 / w
    for i in range(300):
        for j in range(300):
           newImage[i,j] = img[round(i/x), round(j/y)]
    return  newImage

img=cv2.imread("lenna.png")
newimage = nearest(img)
cv2.imshow("old image",img)
cv2.imshow("new image",newimage)
cv2.waitKey(0)
