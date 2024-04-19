import  cv2
import  numpy as np
import matplotlib.pyplot as plt

def linerlinser(img,out_img):
    src_h,src_w,ch=img.shape
    dis_h,dis_w=out_img[1],out_img[0]

    scale_h,scale_w = dis_h / src_h, dis_w / src_h

    if scale_h==1 and scale_w==1:
        return img

    new_img = np.zeros([dis_h,dis_w,ch],dtype=img.dtype)

    for i in range(3):
        for dis_y in range(dis_h):
            for dis_x in range(dis_w):

                #将原图和目标图中心重合

                src_x = (dis_x+ 0.5)/scale_w -0.5
                src_y = (dis_y+ 0.5)/scale_h -0.5

                #算出四个的的坐标
                srcx_0 = int(np.floor(src_x))  # np.floor是向下取整， 就变成3.0了 int（3.0）=3
                srcx_1 = min(int(srcx_0 +1),src_w-1)
                srcy_0 = int(np.floor(src_y))
                srcy_1 = min(int(srcy_0 +1),src_w-1)

                #带入公式 沿着x轴方向做两次单向性插值

                temp0 = (srcx_1-src_x)*img[srcy_0][srcx_0][i] + (src_x-srcx_0)*img[srcy_0][srcx_1][i]
                temp1 = (srcx_1-src_x)*img[srcy_1][srcx_0][i] + (src_x-srcx_0)*img[srcy_1][srcx_1][i]

                #再y轴方向做一次单向性插值

                new_img[dis_y][dis_x][i] = int((srcy_1-src_y)*temp0  +   (src_y-srcy_0)*temp1 )

    return new_img



if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    cv2.imshow("orgin",img)

    size = [700,700]
    result = linerlinser(img,size)

    cv2.imshow("result",result)
    cv2.waitKey(0)


