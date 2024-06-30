import cv2
import numpy as np
import random 

def gs_noise(src, means, sigam, percentage):
    noiseimg = src
    noisenum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(noisenum):
        x = random.randint(0, src.shape[1] - 1)
        y = random.randint(0, src.shape[0] - 1)
        
        noiseimg[x, y] = noiseimg[x, y] + random.gauss(means, sigam)
        
        if noiseimg[x, y] > 255:
            noiseimg[x, y] = 255
        elif noiseimg[x, y] < 0:
            noiseimg[x, y] = 0
        
    return noiseimg
    
    
if __name__ == '__main__':
    img = cv2.imread(r'E:\image\lenna.png', 0)
    img1 = img.copy()
    gsimg = gs_noise(img, 3, 5, 0.9)
    
    cv2.imshow('source', img1)
    cv2.imshow('lenna_gs', gsimg)
    cv2.waitKey(0)