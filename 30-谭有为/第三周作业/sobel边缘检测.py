import  cv2
import  numpy as  np

def img_sobel(img):
    x=cv2.Sobel(img,cv2.CV_16S,1,0)
    y=cv2.Sobel(img,cv2.CV_16S,0,1)
    X=cv2.convertScaleAbs(x)
    Y=cv2.convertScaleAbs(y)
    sobel_img=cv2.addWeighted(X,0.5,Y,0.5,0)
    return  sobel_img

if __name__=='__main__':
    img=cv2.imread('F:/PNG/111.png')
    cv2.imshow('Original image',img)
    img1=img_sobel(img)
    cv2.imshow('sobel image',img1)
    cv2.waitKey()
