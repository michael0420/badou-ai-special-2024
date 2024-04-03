import matplotlib.pyplot as plt
import numpy as np
import cv2


def tianchoutixiang(port):
    h, w, t = port.shape
    empty_huabu = np.zeros((900, 900, t), np.uint8)
    vh = 900 / h
    vw = 900 / w
    for i in range(900):
        for j in range(900):
            x = int(i / vh + 0.5)
            y = int(j / vw + 0.5)
            empty_huabu[i, j] = port[x, y]

    return empty_huabu


port = cv2.imread("lenna.png")
newtu = tianchoutixiang(port)
print(newtu)
print(newtu.shape)
cv2.imshow("nearest interp", newtu)
cv2.imshow("image", port)
cv2.waitKey(0)
