import random
import cv2

def gaussian_noise(source,mean,sigma,percentage):
    noise_source = source
    noise_num = int(noise_source.shape[0]*noise_source.shape[1]*percentage)
    for i in range(noise_num):
        randomX = random.randint(0,source.shape[0]-1)
        randomY = random.randint(0,source.shape[1]-1)
        print(noise_source[randomX,randomY])
        noise_source[randomX,randomY] = noise_source[randomX,randomY]+random.gauss(mean,sigma)
        print(noise_source[randomX,randomY])
        print("")
        if noise_source[randomX,randomY]>255:
            noise_source[randomX, randomY] = 255
        elif noise_source[randomX,randomY]<0:
            noise_source[randomX, randomY] = 0
    return noise_source
if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img1 = gaussian_noise(img_gray,0,0.2,0.8)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gaussian change", img1)
    cv2.imshow("original", img_gray)
    cv2.waitKey(0)
