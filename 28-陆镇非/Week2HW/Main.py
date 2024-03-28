# Author: Zhenfei Lu
# Created Date: 3/27/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com

from Utils import ImageUtils
import cv2
import numpy as np
import time

class Solution(object):
    def __init__(self):
        self.runAlltests()

    def test1(self, imageFilePath: str) -> tuple:
        BGRImage = ImageUtils.readImgFile2BGRImage(imageFilePath)
        greyImage = ImageUtils.BGRImage2GreyImage(BGRImage)
        binaryImage = ImageUtils.greyImage2BinaryImage(greyImage)
        RGBImage = cv2.cvtColor(BGRImage, cv2.COLOR_BGR2RGB)
        dict = {'RGB image': RGBImage, 'grey image' : greyImage, 'binary image': binaryImage}
        ImageUtils.plotAllImages(dict, False)
        return (BGRImage, RGBImage)

    def test2(self, BGRImage: np.ndarray, RGBImage: np.ndarray) -> None:
        target_h = 1024
        target_w = 720
        targetImage = ImageUtils.BGRImageNearestInteropate(BGRImage, np.array([target_h, target_w]))
        (original_h, original_w) = RGBImage.shape[0:2]
        dict2 = {}
        dict2['original RGB image '+str(original_h)+'x'+str(original_w)] = RGBImage
        dict2['nearest interp image '+str(target_h)+'x'+str(target_w)] = targetImage
        ImageUtils.plotAllImages(dict2, False)

    def runAlltests(self):
        # test1
        start_time = time.time()
        filePath = "./lenna.png"
        (BGRImage, RGBImage) = self.test1(filePath)
        end_time = time.time()
        print("test1 excuted time cost：", end_time - start_time, "seconds")

        # test2
        start_time = time.time()
        self.test2(BGRImage, RGBImage)
        end_time = time.time()
        print("test2 excuted time cost：", end_time - start_time, "seconds")

        ImageUtils.showAllPlotsImmediately(True)
        print("All plots shown")

if __name__ == "__main__":
    solution = Solution()
