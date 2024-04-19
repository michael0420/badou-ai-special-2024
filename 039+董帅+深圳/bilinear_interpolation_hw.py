import numpy as np
import cv2
from matplotlib import pyplot as plt

def bilinear_interpolation(ori_img,out_img_size):
    #output original and destinate dimension parameter
    src_w,src_h,channel=ori_img.shape
    dst_w,dst_h=out_img_size
    print('src_w ,src_is :',src_w,src_h)
    print('dst_w,dst_h is :',dst_w,dst_h)
    #if equal, output copied image
    if src_w==dst_w and src_h==dst_h:
        return ori_img.copy()
    #create a new image structure
    dst_img=np.zeros((dst_w,dst_h,3),dtype=np.uint8)
    scale_x=float(src_w/dst_w)
    scale_y=float(src_h/dst_h)
    #for every channel
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_y=dst_y*scale_y-0.5
                src_x=dst_x*scale_x-0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0+1,src_w-1)
                src_y0=int(np.floor(src_y))
                src_y1=min(src_y0+1,src_h-1)

                #for x direction:
                R1=(src_x1-src_x)*ori_img[src_y0,src_x0,i]+(src_x-src_x0)*ori_img[src_y0,src_x1,i]
                R2=(src_x1-src_x)*ori_img[src_y1,src_x0,i]+(src_x-src_x0)*ori_img[src_y1,src_x1,i]
                #FOR Y direction:
                dst_img[dst_y,dst_x,i]=int((src_y1-src_y)*R1+(src_y-src_y0)*R2)

    return dst_img

# if __name__ == '__main__':
#     ori_img=cv2.imread('lenna.png')
#     dts=bilinear_interpolation(ori_img,(700,700))
#     cv2.imshow('bilinear',dts)
#     cv2.imshow('original',ori_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     #read the image and covert it to grayscale
#     ori2_img=cv2.imread('lenna.png')
#     gray_img=cv2.cvtColor(ori2_img,cv2.COLOR_BGR2GRAY)
#     cv2.imshow('gray image',gray_img)
#     dst_img = cv2.equalizeHist(gray_img)
#     cv2.imshow('dst image', dst_img)
#     cv2.waitKey(5000)
#     cv2.destroyAllWindows()
#     #calculate the histogram
#     hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
#     plt.plot(hist,color='black')
#     plt.xlabel('pixel intensity')
#     plt.ylabel('Frequency')
#     plt.title('Grayscale Histogram')
#     plt.show()


# if __name__ == '__main__':
#     img=cv2.imread('lenna.png',1)
#     if img is None:
#         print('error')
#     else:
#     # Split the image into its channels
#     (b,g,r)=cv2.split(img)
#     # Equalize histogram for each channel
#     b_hist =cv2.equalizeHist(b)
#     g_hist =cv2.equalizeHist(g)
#     r_hist=cv2.equalizeHist(r)
#     # Merge the equalized channels into a color image
#     result=cv2.merge([b_hist,g_hist,r_hist])
#     cv2.imshow('dst',result)
#     cv2.waitKey(5000)
#     cv2.destroyAllWindows()

if __name__ == '__main__':
    ori_img=cv2.imread('lenna.png')
    x=cv2.Sobel(ori_img,cv2.CV_16S,1,0)
    y=cv2.Sobel(ori_img,cv2.CV_16S,0,1)

    absX = cv2.convertScaleAbs(x)
    absY=cv2.convertScaleAbs(y)
    dst=cv2.addWeighted(absX,0.5,absY,0.5,0)

    cv2.imshow('absX',absX)
    cv2.imshow('absY',absY)
    cv2.imshow('dst',dst)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()












