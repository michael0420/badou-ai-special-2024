import cv2
import os

if __name__ == "__main__":
    # 设置图片路径
    base_dir = "img"
    img1_name , img2_name = "iphone1.png" , "iphone2.png"
    img1_path = os.path.join(base_dir, img1_name)
    img2_path = os.path.join(base_dir, img2_name)

    # 读取图片
    img1 = cv2.imread(img1_path)  # 查询图片
    img2 = cv2.imread(img2_path)  # 训练图片

    # 初始化SIFT检测器
    sift = cv2.xfeatures2d.SIFT_create()

    # 使用SIFT找到关键点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 创建BFMatcher对象
    bf = cv2.BFMatcher()

    # 使用KNN算法进行匹配
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 应用比率测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.35 * n.distance:
            good_matches.append([m])

    # 绘制匹配结果
    img_matches = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)

    # 显示结果
    cv2.imshow('SIFT Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()