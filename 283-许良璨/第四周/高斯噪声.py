import cv2
import random

def GaussianNoise(src,means,sigma,SNR):
    dst=src
    h,w=src.shape[:2]
    totaln=int(h*w*SNR)
    for i in range(totaln):
        x=random.randint(0,w-1)
        y=random.randint(0,h-1)
        dst[y,x]=dst[y,x]+random.gauss(means,sigma)
        if  dst[y,x]<0:
            dst[y, x]=0
        elif  dst[y,x]>255:
            dst[y, x]=255
    return dst

if __name__ == '__main__':
     src = cv2.imread("../lenna.png",0)
     img1 = GaussianNoise(src, 2, 4, 0.8)
     img2 = GaussianNoise(src, 2, 10, 0.8)
     cv2.imshow("src", src)
     cv2.imshow("img1",img1)
     cv2.imshow("img2", img2)
     cv2.waitKey(0)
     cv2.destroyAllWindows()