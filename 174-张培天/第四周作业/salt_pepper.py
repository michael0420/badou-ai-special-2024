import cv2
import random

def saltPepperNoise(src, percetage):
    noiseImg = src
    noiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)

        if random.random() <= 0.5:
            noiseImg[randX, randY] = 0
        else:
            noiseImg[randX, randY] = 255
    return noiseImg

if __name__ == "__main__":
    img = cv2.imread("lenna.png", 0)
    img_1 = saltPepperNoise(img, 0.15)
    img = cv2.imread(r"E:\data\dabou\lenna.png")
    img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('source img', img_2)
    cv2.imshow('salt pepper nosie img', img_1)
    cv2.waitKey(0)