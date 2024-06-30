import cv2
import random

def GaussianNoise(src, mean, sigma, percetage):
    nosieImg = src
    nosieNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(nosieNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)

        nosieImg[randX, randY] += random.gauss(mean, sigma)
        if nosieImg[randX, randY] < 0:
            nosieImg[randX, randY] = 0
        elif nosieImg[randX, randY] > 255:
            nosieImg[randX, randY] = 255
    return nosieImg

if __name__ == "__main__":
    img = cv2.imread("enna.png", 0)
    img_1 = GaussianNoise(img, 2, 4, 0.8)
    img = cv2.imread(r"E:\data\dabou\lenna.png")
    img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('source img', img_2)
    cv2.imshow('GaussionNosie img', img_1)
    cv2.waitKey(0)
