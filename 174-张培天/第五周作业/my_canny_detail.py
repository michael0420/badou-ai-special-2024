import numpy as np
import matplotlib.pyplot as plt
import math

def _Gaussian(img,sigma,dim):
    Gaussian_filter = np.zeros([dim,dim])
    tmp = [i - dim // 2 for i in range(dim)]
    n1= 1 / (2 * math.pi * sigma ** 2)
    n2 = - 1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    img_new = np.zeros(img.shape)
    tmp = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim] * Gaussian_filter)
    return img_new

def _Tidu(img):
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img.shape)
    img_tidu_y = np.zeros(img.shape)
    img_tidu = np.zeros(img.shape)
    img_pad = np.pad(img, ((1,1), (1,1)), 'constant')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_kernel_y)
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y / img_tidu_x
    return img_tidu, angle

def _NotBigYiZhi(img, angle):
    img_yizhi = np.zeros(img.shape)
    for i in range(1, img.shape[0] -1):
        for j in range(1, img.shape[1] -1):
            flag = True
            temp = img[i-1:i+2, j-1:j+2]
            if angle[i,j] <= -1:
                num_1 = (temp[0,1] - temp[0,0]) / angle[i,j] + temp[0,1]
                num_2 = (temp[2,1] - temp[2,2]) / angle[i,j] + temp[2,1]
                if not (img[i,j] > num_1 and img[i,j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img[i, j] > num_1 and img[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img[i, j] > num_1 and img[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img[i, j] > num_1 and img[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i,j] = img[i,j]
    return img_yizhi

def _Boundary(img,tidu_img):
    lower_boundary = tidu_img.mean() * 0.5
    high_boundary = lower_boundary * 3
    zhan = []
    for i in range(1, img.shape[0] -1):
        for j in range(1, img.shape[1] -1):
            if img[i,j] >= high_boundary:
                img[i,j] = 255
                zhan.append([i,j])
            elif img[i,j] <= lower_boundary:
                img[i,j] = 0
    
    while zhan:
        temp_1, temp_2 = zhan.pop()
        a = img[temp_1-1:temp_1+2,temp_2-1:temp_2+2]
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if i == 1 and j == 1:
                    continue
                if (a[i,j] < high_boundary) and (a[i,j] > lower_boundary):
                    img[temp_1-1+i, temp_2-1+j] = 255
                    zhan.append([temp_1-1+i, temp_2-1+j])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] !=0 and img[i,j] != 255:
                img[i,j] = 0
    return img

def _myCanny(img,sigma,dim):
    gaussian_img = _Gaussian(img, sigma, dim)
    tidu_img, angle = _Tidu(gaussian_img)
    yizhi_img = _NotBigYiZhi(tidu_img,angle)
    canny_img = _Boundary(yizhi_img, tidu_img)
    plt.figure(1)
    plt.imshow(gaussian_img.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.figure(2)
    plt.imshow(tidu_img.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.figure(3)
    plt.imshow(yizhi_img.astype(np.uint8), cmap='gray')
    plt.axis('off')  
    plt.figure(4)
    plt.imshow(canny_img.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()    

if __name__ == "__main__":
    pic_path = "lenna.png"
    sigma = 0.5
    dim = 5
    img = plt.imread(pic_path)
    # print("image", img)
    if pic_path.endswith(".png"):
        img = img * 255
    img = img.mean(axis=-1)
    _myCanny(img,sigma,dim)