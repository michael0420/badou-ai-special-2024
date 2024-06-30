# _*_ coding: UTF-8 _*_
# @Time: 2024/5/23 19:57
# @Author: iris
# @Email: liuhw0225@126.com
import cv2

if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(gray, None)

    img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            color=(51, 163, 236))

    cv2.imshow('sift_keypoints', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
