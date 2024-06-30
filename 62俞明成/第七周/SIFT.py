import cv2 as cv
import numpy as np


class SIFT:

    def __init__(self):
        self.sift = cv.SIFT.create()

    def detect_draw(self, detect_img):
        """
        关键点检测
        :param detect_img: 被检测的图像
        :return: 返回检测后对每个关键点都绘制了圆圈和方向的图像
        """
        detect_img_gray = cv.cvtColor(detect_img, cv.COLOR_RGB2GRAY)
        # 检测图像中的关键点
        key_points, descriptor = self.sift.detectAndCompute(detect_img_gray, None)
        # 对图像的每个关键点都绘制了圆圈和方向
        result_img = cv.drawKeypoints(image=detect_img, outImage=detect_img, keypoints=key_points,
                                      flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                      color=(51, 163, 236))
        return result_img

    def show_detect_img(self, img):
        result = self.detect_draw(img)
        cv.imshow('sift_key_points', result)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def match(self, img1, img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        bf = cv.BFMatcher(cv.NORM_L2)
        # opencv中knnMatch是一种蛮力匹配
        # 将待匹配图片的特征与目标图片中的全部特征全量遍历，找出相似度最高的前k个。
        matches = bf.knnMatch(des1, des2, k=2)

        # 筛选好的匹配
        good_match = []
        for m, n in matches:
            if m.distance < 0.50 * n.distance:
                # good_match.append([m])
                good_match.append(m)
        # print(good_match)
        return self.drawMatchesKnn_cv2(img1, kp1, img2, kp2, good_match)
        # result = cv.drawMatchesKnn(img1, kp1, img2, kp2, good_match, None,
        #                            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imshow("match", result)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    def drawMatchesKnn_cv2(self, img1_gray, kp1, img2_gray, kp2, goodMatch):
        h1, w1 = img1_gray.shape[:2]
        h2, w2 = img2_gray.shape[:2]

        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        # 拼接
        vis[:h1, :w1] = img1_gray
        vis[:h2, w1:w1 + w2] = img2_gray
        # 提取匹配的特征点索引
        p1 = [kpp.queryIdx for kpp in goodMatch]
        p2 = [kpp.trainIdx for kpp in goodMatch]

        # 提取匹配的特征点位置并转换坐标
        post1 = np.int32([kp1[pp].pt for pp in p1])
        post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

        # 绘制连线
        for (x1, y1), (x2, y2) in zip(post1, post2):
            cv.line(vis, (x1, y1), (x2, y2), (0, 0, 255))

        cv.namedWindow("match", cv.WINDOW_NORMAL)
        cv.imshow("match", vis)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    img = cv.imread("../lenna.png")
    img1 = cv.imread("../iphone1.png")
    img2 = cv.imread("../iphone2.png")
    sift = SIFT()
    sift.show_detect_img(img)
    sift.match(img1, img2)
