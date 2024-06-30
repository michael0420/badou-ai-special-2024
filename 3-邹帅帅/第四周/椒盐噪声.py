import cv2
import numpy as np
import random 

def jy_noise(src, percentage):
    noiseimg = src
    noisenum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(noisenum):
        x = random.randint(0, src.shape[1] - 1)
        y = random.randint(0, src.shape[0] - 1)
        if random.random() <= 0.5:
            noiseimg[x, y] = 0
        else:
            noiseimg[x, y] = 255
            
    return noiseimg
    
    
if __name__ == '__main__':
    img = cv2.imread(r'E:\image\lenna.png', 0)
    img1 = img.copy()
    jyimg = jy_noise(img, 0.05)
    
    cv2.imshow('source', img1)
    cv2.imshow('lenna_jy', jyimg)
    cv2.waitKey(0)