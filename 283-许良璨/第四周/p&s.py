import cv2
import random

def psNoise (img,SNR):
    dst=img
    h,w=img.shape[:2]
    totalN=int(h*w*SNR)
    for i in range(totalN):
        x=random.randint(0,w-1)
        y=random.randint(0,h-1)
        if random.random()<0.5:
            dst[y,x]=0
        else:
            dst[y,x]=255
    return dst

if __name__=="__main__":
    img = cv2.imread("../lenna.png")
    img1=psNoise(img,0.1)
    cv2.imshow("img1",img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
