import numpy as np  # 导入numpy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，用于绘图
import math  # 导入math库，用于数学计算
import cv2

# 设置打印格式
np.set_printoptions(precision=5, suppress=True, linewidth=1000)
# precision控制打印小数位数
# suppress=True可以禁止科学计数法的打印格式
# linewidth=1000表示单行打印的长度,设置足够大可以完整打印

# np.set_printoptions(threshold=np.inf)  # 设置打印选项，将阈值设置为正无穷，以完整显示整个数组

if __name__ == '__main__':  # 如果直接运行该脚本，则执行以下代码
    pic_path = 'lenna.png'  # 设置图片路径
    img = plt.imread(pic_path)  # 读取图片
    # plt.imshow(img)  # 展示正常
    # plt.show()

    # 颜色通道顺序：
    # plt.imread() 读取的图片颜色通道顺序通常是 RGB。
    # cv.imread() 读取的图片颜色通道顺序通常是 BGR，这意味着蓝色和红色通道的位置与 plt.imread() 读取的图片相反。

    # 透明度支持：
    # plt.imread() 在读取 PNG 图像时能够保留透明度通道。
    # cv.imread() 默认情况下不保留透明度通道。如果需要，可以使用 cv.IMREAD_UNCHANGED 标志来读取包含透明度通道的图像。

    print("img：\n", img, f"\n{len(img)}")  # 打印图片数据
    if pic_path[-4:] == '.png':
        # cv.imread() 总是读取图片到 uint8 类型，并且值在 [0, 255] 范围内。
        # 如果图片格式为png, png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型 将图片数据乘以255，因为png图片数据范围为0到1
        # plt.imshow(img)
        # plt.show()
        print("img--:\n", img, f"\n{len(img)}")

    img = img.mean(axis=-1)  # 对图片数据每个数组到最后一个维度求均值，实现灰度化  每个数组就剩下一个值  多个三通道变成多个单通道
    # plt.imshow(img)
    # plt.show()
    print("imgmean：\n", img, f"\n{len(img)}")  # 打印图片数据
    '''
    这个代码进行img = img.mean(axis=-1)处理的目的是实现图像的灰度化。
    原始读取进来的图片数据img是三通道的RGB颜色空间,每个像素点由三个通道值组成,比如(R,G,B),把这里的RGB的值求均值然后每个RGB三个值变成一个值转灰度化
    而灰度图像只有一个通道,每个像素点用一个值表示亮度。
    这里通过img.mean(axis=-1)将每个像素点三个通道的值进行平均,求得这个像素点的亮度值,这样就实现了将RGB图像转换为单通道的灰度图像。
    轴向-1表示对img数组最后一个维度(通道维度)求平均。具体来说,对每个像素点的R、G、B三个值取平均,就得到这个像素点在灰度图像中的唯一值,实现了颜色图像到灰度图像的转换。
    这样做的目的是为后续Canny边缘检测算法提供单通道的输入图像。Canny算法本身只适用于灰度图像,需要先将RGB图像预处理为灰度空间。
    所以总之,img = img.mean(axis=-1)这步是为了实现图像的灰度化预处理,为后续Canny边缘检测算法提供适当的单通道输入图像。    
    '''

    # # opencv展示
    # mg = cv2.imread(pic_path)  # 读取图片  PNG 要么转为jpg 不然就是蓝色的调调
    # img_rgb = cv2.cvtColor(mg, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB  否则直接读取plt.imread()读取过的图片会调返蓝色重
    # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)   # Create a window to display the image
    # cv2.imshow('Image', img_rgb)              # Display the image in the window
    # cv2.waitKey(0)                        # Wait for a key press to close the window:
    # cv2.destroyAllWindows()               # Close the window and clean up

    # 1、高斯平滑
    # sigma = 1.52  # 高斯平滑时的高斯核参数，标准差，可调
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    dim = 5  # 高斯核尺寸
    Gaussian_filter = np.zeros([dim, dim])  # 创建一个5x5的零矩阵，用于存储高斯核，这是数组不是列表了
    print("Gaussian_filter:\n", Gaussian_filter, f'{len(Gaussian_filter)}')
    '''
    Gaussian_filter:
     [[0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]] 5
    '''
    # plt.title('Gaussian_filter')
    # plt.imshow(Gaussian_filter)
    # plt.show()
    # # plt.imshow(Gaussian_filter, cmap='gray')   # 强制指定灰度图
    # # plt.show()

    '''
    在使用plt.imshow()函数显示图像时，对于大多数情况而言，不需要将图像数据显式地转换为uint8类型。matplotlib库的imshow()函数能够自动处理不同类型的图像数据。
    plt.imshow()函数在接收图像数据时，会自动根据数据的类型进行处理和调整。它可以接受多种数据类型的输入，包括uint8、float32等。
    在这种情况下，如果Gaussian_filter的数据类型是float64或float32，plt.imshow()函数仍然能够正确地显示图像。因此，对于Gaussian_filter这样的小型矩阵，不进行显式的数据类型转换也是可以的。
    需要注意的是，当图像数据是大型图像或者数据类型不是uint8时，可能需要根据情况进行数据类型转换，以便正确地显示图像。
    '''

    # # 将图像数据转换为uint8类型
    # Gaussian_filter = Gaussian_filter.astype(np.uint8)
    # cv2.imshow("Gaussian_filter", Gaussian_filter)  # 显示图片
    # cv2.waitKey(0)  # 等待按键
    # cv2.destroyAllWindows()  # 关闭窗口
    '''
    在使用OpenCV库中的`cv2.imshow()`函数显示图像时，图像数据需要满足特定的数据类型要求。通常情况下，图像数据的类型应为`uint8`（无符号8位整数）。
    将图像数据转换为`uint8`类型有以下几个原因：
    1. **数据范围限制：** `uint8`类型的数据范围是0到255，正好适用于表示图像的灰度值或颜色通道值。通过将数据转换为`uint8`类型，可以确保图像数据的数值范围正确，并符合OpenCV函数对数据范围的要求。
    2. **内存占用：** `uint8`类型占用的内存空间较小，每个像素只需要一个字节进行存储。相对于其他数据类型（如`float32`），使用`uint8`类型可以减少图像数据在内存中占用的空间，提高内存的利用效率。
    3. **显示要求：** `cv2.imshow()`函数要求图像数据为`uint8`类型，以便正确显示图像。如果图像数据不符合要求，可能会导致显示错误或异常。
    因此，为了满足OpenCV的要求并确保正确的图像显示，需要将图像数据转换为`uint8`类型。

    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    在Python中使用matplotlib的`imshow`函数和OpenCV的`imshow`函数显示图像时，两者对数据类型的处理和显示方式有所不同，这可能是导致你观察到不同颜色的原因。

    1. **matplotlib的imshow函数**：
       - 当使用matplotlib的`imshow`函数显示图像时，它通常能够自动处理不同类型的图像数据，包括浮点数类型的数据。
       - 如果图像数据是浮点数类型，matplotlib会将其缩放到0-1的范围（如果数据已经在这个范围内）。
       - 在你的代码中，`Gaussian_filter`初始化为一个全零的浮点数矩阵。理论上，这应该显示为黑色的图像，因为所有的像素值都是0，对应于黑色。

    2. **OpenCV的imshow函数**：
       - OpenCV的`imshow`函数通常要求图像数据是8位无符号整数类型（即`uint8`），范围从0到255。
       - 当你使用`astype(np.uint8)`将`Gaussian_filter`转换为`uint8`类型时，你实际上是在告诉OpenCV这是一个有效的灰度图像，其中所有的值都等于0，这将被正确地显示为黑色。

    现在，关于为什么matplotlib显示的是紫色而不是黑色，这可能是由于以下原因之一：

    - **颜色映射（Colormap）**：matplotlib默认使用一种颜色映射，当显示浮点数类型的图像数据时，如果数据值为0，它可能不会显示为纯黑色，而是颜色映射中的一个特定颜色，这可能是紫色或其他颜色。

    - **数据类型问题**：如果`Gaussian_filter`矩阵中的数据类型不是纯浮点数类型，或者存在某种精度问题，这可能会影响显示结果。

    - **matplotlib的bug或特性**：在某些情况下，matplotlib可能由于内部处理方式的不同，导致显示的颜色与预期不符。

    为了解决这个问题，你可以尝试以下步骤：

    - 确保`Gaussian_filter`矩阵中的数据类型是纯浮点数类型，并且所有值都是0。
    - 在使用matplotlib显示图像之前，检查并确认`Gaussian_filter`的数据类型和值。
    - 如果问题仍然存在，尝试使用不同的颜色映射，例如使用`plt.imshow(Gaussian_filter, cmap='gray')`来强制使用灰度颜色映射。

    最后，你的代码中有一行注释提到了不需要将图像数据显式地转换为uint8类型，这是正确的。但是，如果你希望确保图像以预期的方式显示，尤其是在不同的库之间切换时，进行数据类型转换通常是一个好的实践。
    '''

    # 通过生成卷积核序列tmp并计算高斯核的系数和指数部分，就可以得到完整的高斯核
    tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列，用于计算高斯核
    print("tmp:\n", tmp)  # [-2, -1, 0, 1, 2]   这个序列的目的是生成一个以0为中心的对称序列，用于计算高斯核的各个位置的权重。

    # 生成高斯公式计算
    # n1 高斯核的系数部分 它与高斯函数的归一化有关，确保当所有元素乘以对应的高斯核值后，核的总和为1（或者在某些情况下，总和为一个特定的值，比如图像亮度的均值）。
    n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核的系数   math.pi 是圆周率Π
    print("n1:\n", n1)  # 0.6366197723675814
    # print(math.pi)    # 3.141592653589793  圆周率

    # n2 是高斯核的指数部分，它与高斯函数的形状有关：
    n2 = - 1 / (2 * sigma ** 2)  # 高斯核的指数  也是高斯公式的分母要乘于的e的幂的系数的次方部分
    print("n2:\n", n2)  # -2.0

    for i in range(dim):  # 计算高斯核的值
        for j in range(dim):
            # σ (sigma) 是高斯核的标准差，控制着高斯函数的宽度（在图像处理中，也称为尺度或尺度参数）。
            # e (math.exp计算e的幂，e的函数真math.e 返回一个e的近似值) 是自然对数的底数，约等于 2.71828。它决定了高斯曲线的下降速度，也就是高斯核的宽度
            # 每一个像素点都做高斯卷积核的高斯公式计算
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))

    # 得到了最终的高斯核，可以用于图像的平滑处理    其实分子的Gaussian_filter已经是高斯核的样子了
    # print(Gaussian_filter)
    # print("Gaussian_filter:\n", Gaussian_filter, f'{len(Gaussian_filter)}')
    # print("Gaussian_filter_sum:\n",Gaussian_filter.sum())   #  1.0289743877169395

    # 把矩阵的每一个数字都 / sum总和   这样子算出来的矩阵加总就是1
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()  # 归一化高斯核  为了确保高斯核的总和为1（归一化）
    print("Gaussian_filter:\n", Gaussian_filter, f'{len(Gaussian_filter)}')
    # print(Gaussian_filter.sum())   # 1.0000000000000002
    # plt.imshow(Gaussian_filter)   # 5*5 高斯核图片
    # plt.show()

    dx, dy = img.shape  # 获取原图像的尺寸  拆包
    print(img.shape)  # (512, 512)
    # print("dx:\n", dx)
    # print("dy:\n", dy)

    img_new = np.zeros(img.shape)  # 创建一个与原图像同尺寸的零矩阵, 存储平滑之后的图像，zeros函数得到的是浮点型数据
    # plt.title('img_new')
    # plt.imshow(img_new)
    # plt.axis('off')
    # plt.show()

    tmp = dim // 2
    # 对图像进四周行边缘填充 tmp个像素    填充的方式是 'constant'，意味着会用一个常数值（默认为0）来填充新创建的边缘
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
    # img_pad_uint8 = np.clip(img_pad, 0, 255).astype(np.uint8)
    # plt.imshow(img_pad_uint8)
    # plt.imshow(img_pad)
    # plt.title('img_pad')
    # plt.axis('off')  # 取消坐标轴
    # plt.show()
    print("img_pad:\n", img_pad,len(img_pad))

    # 高斯卷积核去卷
    for i in range(dx):  # 对图像进行高斯平滑
        for j in range(dy):
            # 因为卷积核是5*5 卷完的图像是外周比原图两个像素
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)

    print("img_new:\n", img_new,len(img_new))

    # plt.figure(1)
    # # plt.imshow(img_new)
    # # # 显示平滑后的图像，并将数据类型转换为uint8
    # # plt.imshow(img_new.astype(np.uint8))    # 伪彩图 这里展示的还是蓝色 只不过是蓝色的图片然后高斯模糊过了
    # # plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶，其实不转也可以
    # plt.imshow(img_new, cmap='gray')  # 使用灰度颜色映射
    # plt.title('img_new')
    # plt.axis('off')  # 取消坐标轴
    # plt.show()

    # cv2.namedWindow('img_new', cv2.WINDOW_NORMAL)   # Create a window to display the image
    # cv2.imshow('img_new', img_new)              # Display the image in the window
    # cv2.waitKey(0)                        # Wait for a key press to close the window:
    # cv2.destroyAllWindows()               # Close the window and clean up
    '''
    img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
    ```
    
    正在进行的是图像的高斯平滑处理，这是通过将高斯核（`Gaussian_filter`）应用于图像（`img_pad`）的每个部分来完成的。具体来说，这行代码执行以下操作：
    
    1. `img_pad[i:i + dim, j:j + dim]`：这部分是从图像 `img_pad` 中提取一个以当前处理的像素点 `(i, j)` 为中心，大小为 `(dim, dim)` 的局部区域（也称为窗口或块）。
    `dim` 是高斯核的尺寸，通常是一个奇数，比如 3、5 或 7，这样核才有精确的中心。
    这个表达式是用来从图像 img_pad 中截取一个以像素点 (i, j) 为中心的局部区域（邻域）。dim 是高斯核的尺寸，所以这个切片取出了一个 dim x dim 的图像块。

    具体来说：
    i:i + dim 表示从第 i 个元素开始取到第 i + dim 个元素（不包括）的一系列整数。如果 dim 是 5，那么这将取出从 i 到 i + 4 的整数（例如，如果 i 是 2，那么取出的是 2, 3, 4, 5, 6）。
    j:j + dim 的工作方式相同，但它是应用于列索引的。
    将它们结合起来，img_pad[i:i + dim, j:j + dim] 就取出了图像 img_pad 中的一个 dim x dim 的块，这个块以 (i, j) 为中心。
    在图像处理的上下文中，这个操作通常用于以下目的：
    将核（或滤波器、卷积核）应用于图像的每个局部区域。
    计算核与图像块内像素值的加权和，这通常用于生成平滑图像或应用其他滤波效果。
    例如，如果您正在使用一个 3x3 的高斯核对图像进行平滑处理，那么您会对图像中的每个 3x3 像素块应用这个核。切片 img_pad[i:i + 3, j:j + 3] 正是为了这个目的，它取出了当前处理的像素及其八个邻居，以便计算它们的加权平均值。

    2. `*`：这是元素乘法操作符，它将高斯核的每个元素与其对应的图像局部区域的元素相乘。也就是说，核中的每个权重会乘以图像中相应位置的像素值。
    
    3. `np.sum(...)`：这是一个函数，计算上述元素乘法结果的总和。这个总和就是应用高斯核后在当前像素位置 `(i, j)` 的结果值。
    
    4. `img_new[i, j]`：将上述计算得到的总和赋值给新的图像矩阵 `img_new` 中的对应像素位置 `(i, j)`。
    
    综合起来，这行代码的作用是：对于图像 `img_pad` 的每个像素点 `(i, j)`，用高斯核 `Gaussian_filter` 对其周围的局部区域进行加权求和，然后将这个加权求和的结果作为新图像 `img_new` 在 `(i, j)` 位置的像素值。
    
    这个过程对图像 `img_pad` 中的每个像素点重复执行，从而生成一个新的图像 `img_new`，其中已经应用了高斯平滑处理。
    高斯平滑有助于减少图像噪声，因为核中心以外的像素（即图像的边缘）会被乘以较小的权重，而中心像素则有较大的权重。
    
    在实际应用中，这一步通常用于边缘检测算法之前，以减少噪声对边缘检测结果的影响。
    '''

    # 2、求梯度。以下两个是滤波用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel算子，用于计算水平梯度
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Sobel算子，用于计算垂直梯度

    img_tidu_x = np.zeros(img_new.shape)  # 创建一个与平滑后图像同尺寸的零矩阵，用于存储水平梯度
    img_tidu_y = np.zeros([dx, dy])  # 创建一个与平滑后图像同尺寸的零矩阵，用于存储垂直梯度
    img_tidu = np.zeros(img_new.shape)  # 创建一个与平滑后图像同尺寸的零矩阵，用于存储梯度幅值
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 对平滑后的图像进行边缘填充，根据上面矩阵结构所以写1

    for i in range(dx):  # 计算梯度
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # 计算水平梯度
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # 计算垂直梯度
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)  # 计算梯度幅值 上面两个的平方相加开根号

    '''
    在 NumPy 中，这种操作被称为布尔索引或掩码索引，它允许你根据一个布尔条件来选择数组的元素，并只对这些选中的元素进行操作。
    具体来说，表达式 img_tidu_x == 0 会创建一个布尔数组，该数组中的每个元素都是 True 或 False，取决于 img_tidu_x 中相应位置的元素是否等于零。
    然后，使用这个布尔数组进行索引，img_tidu_x[img_tidu_x == 0] 会选择 img_tidu_x 中所有值为零的元素。只有这些被选中的元素（即那些位置为 True 的元素）的值会被后续的赋值操作 = 0.00000001 改变。
    这意味着：
    当 img_tidu_x 中的元素等于 0 时，它们会被设置为 0.00000001。
    当 img_tidu_x 中的元素不等于 0 时，它们保持不变。
    这是一种非常高效的方式来更新数组中满足特定条件的元素。这种方式在数值计算和图像处理中非常常见，因为它可以避免对数组中的所有元素进行循环，而是利用 NumPy 的向量化操作来快速完成这些任务。
    '''
    img_tidu_x[img_tidu_x == 0] = 0.00000001  # 避免除数为零
    angle = img_tidu_y / img_tidu_x  # 计算梯度方向

    print("img_tidu_x:\n",img_tidu_x)
    print("img_tidu_y:\n",img_tidu_y)
    print("img_tidu:\n",img_tidu)
    print("angle:\n",angle)

    # # plt.figure(2)
    # # plt.imshow(img_tidu_x.astype(np.uint8), cmap='gray')  # 显示梯度幅值图像
    # # plt.axis('off')
    # # plt.show()
    #
    # # plt.figure(2)
    # # plt.imshow(img_tidu_y.astype(np.uint8), cmap='gray')  # 显示梯度幅值图像
    # # plt.axis('off')
    # # plt.show()
    #
    # plt.figure(2)
    # plt.imshow(img_tidu.astype(np.uint8), cmap='gray')  # 显示梯度幅值图像
    # plt.axis('off')
    # plt.show()
    #
    # # plt.figure(2)
    # # plt.imshow(angle.astype(np.uint8), cmap='gray')  # 显示梯度幅值图像
    # # plt.axis('off')
    # # plt.show()

    # 3、非极大值抑制 （Non-Maximum Suppression, NMS）
    '''
    非极大值抑制（Non-Maximum Suppression, NMS）是Canny边缘检测算法中的一个关键步骤，它的作用是在确定了梯度幅值和梯度方向后，进一步精细化边缘，去除那些不是边缘的像素点。
    让我们逐步分析这段代码：
    1. `img_yizhi = np.zeros(img_tidu.shape)`：创建一个新的数组 `img_yizhi`，它的大小与梯度幅值图像 `img_tidu` 相同，并且初始化为零。这个数组将用来存储非极大值抑制后的结果。
    2. `for i in range(1, dx - 1): for j in range(1, dy - 1):`：这两个嵌套循环遍历 `img_tidu` 中的每个像素点，但不包括边缘像素（即第一行和第一列，以及最后一行和最后一列），因为这些像素的邻域不完整。
    3. `flag = True`：初始化一个标志变量 `flag`，用来标记当前像素点是否应该被保留（即是否是边缘点）。
    4. `temp = img_tidu[i - 1:i + 2, j - 1:j + 2]`：获取当前像素点 `(i, j)` 的3x3邻域，这个邻域包含了当前像素点及其周围的8个像素点。
    5. 接下来的 `if-elif` 块根据梯度方向 `angle[i, j]` 算出来的值，计算两个线性插值的数值 `num_1` 和 `num_2`。这些数值用于比较当前像素点的梯度幅值，以确定它是否是边缘点。
    6. `if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):`：如果当前像素点的梯度幅值不大于其在梯度方向上的两个线性插值数值，则认为它不是边缘点，将 `flag` 设置为 `False`。
    7. `if flag: img_yizhi[i, j] = img_tidu[i, j]`：如果 `flag` 仍然为 `True`（即当前像素点被认为是边缘点），则将 `img_tidu` 中该像素点的梯度幅值复制到 `img_yizhi` 中。
    非极大值抑制的目的是确保边缘是沿着梯度方向的局部最大值。具体来说，对于每个像素点，算法会检查其梯度方向上的两个相邻像素点的梯度幅值。如果当前像素点的梯度幅值大于这两个相邻点的梯度幅值，则保留该点；否则，抑制该点（即将其设置为零）。
    这个过程最终会在 `img_yizhi` 中留下清晰的边缘，其中边缘是沿着梯度方向的局部最大值。这些边缘将通过后续的双阈值检测步骤进一步处理，以确定最终的边缘像素。
    '''
    # 创建一个与梯度幅值图像同尺寸的零矩阵，用于存储非极大值抑制后的图像
    img_yizhi = np.zeros(img_tidu.shape)  # img_yizhi：保存非极大值抑制后的图像，初始化为全零矩阵

    # 对梯度幅值图像进行非极大值抑制，遍历每个像素点（除去图像边缘）
    for i in range(1, dx - 1):  # i：行索引，从1开始到dx-2，避免边缘
        for j in range(1, dy - 1):  # j：列索引，从1开始到dy-2，避免边缘
            flag = True  # 标记是否需要抑制该点，初始设为True
            # 获取梯度幅值的8邻域矩阵
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # temp：当前像素点的3x3邻域矩阵

            # 使用线性插值法判断是否进行非极大值抑制
            if angle[i, j] <= -1:  # 如果梯度方向角度 <= -1
                # 计算插值点的梯度值
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]  # num_1：插值计算得到的梯度值1
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]  # num_2：插值计算得到的梯度值2
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):  # 如果当前点梯度值不大于插值点梯度值
                    flag = False  # 标记为False，表示该点需要抑制

            elif angle[i, j] >= 1:  # 如果梯度方向角度 >= 1
                # 计算插值点的梯度值
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]  # num_1：插值计算得到的梯度值1
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]  # num_2：插值计算得到的梯度值2
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):  # 如果当前点梯度值不大于插值点梯度值
                    flag = False  # 标记为False，表示该点需要抑制

            elif angle[i, j] > 0:  # 如果梯度方向角度在0和1之间
                # 计算插值点的梯度值
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]  # num_1：插值计算得到的梯度值1
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]  # num_2：插值计算得到的梯度值2
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):  # 如果当前点梯度值不大于插值点梯度值
                    flag = False  # 标记为False，表示该点需要抑制

            elif angle[i, j] < 0:  # 如果梯度方向角度在-1和0之间
                # 计算插值点的梯度值
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]  # num_1：插值计算得到的梯度值1
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]  # num_2：插值计算得到的梯度值2
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):  # 如果当前点梯度值不大于插值点梯度值
                    flag = False  # 标记为False，表示该点需要抑制

            # 如果需要保留该点，则将梯度赋值更新后的像素点其值赋给非极大值抑制后的图像
            if flag:  # 如果flag为True，表示该点不需要抑制
                img_yizhi[i, j] = img_tidu[i, j]  # 将当前点梯度值赋给非极大值抑制后的图像对应位置
            '''
            这段代码主要实现了 Canny 边缘检测算法中的非极大值抑制部分。下面是逐行详细的中文注释：
            ### 代码逻辑和作用
            1. **初始化**：创建一个与梯度幅值图像（`img_tidu`）相同尺寸的零矩阵`img_yizhi`，用于存储非极大值抑制后的图像。
            2. **遍历图像**：对梯度幅值图像的每个像素点进行遍历（边缘像素除外）。
            3. **局部邻域处理**：获取当前像素点的3x3邻域矩阵，通过线性插值法计算梯度方向上的插值点梯度值。
            4. **抑制判断**：根据插值点梯度值与当前点梯度值的大小关系，决定是否抑制当前点。如果当前点的梯度值不大于插值点的梯度值，则标记为需要抑制。
            5. **更新输出图像**：如果当前点不需要抑制，则将其梯度值赋给输出图像`img_yizhi`对应位置。
            **总结**：这段代码实现了 Canny 边缘检测中的非极大值抑制步骤，通过对梯度幅值图像进行处理，保留潜在的边缘点，抑制非边缘点，从而在图像中提取更准确的边缘信息。
            '''
    # plt.figure(3)
    # plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')  # 显示非极大值抑制后的图像
    # plt.axis('off')
    # plt.show()
    print("img_yizhi:\n",img_yizhi,len(img_yizhi),img_yizhi.shape)
    
    '''
    这段代码是Canny边缘检测算法中的双阈值检测和边缘连接步骤。让我们逐步解释这个过程：

     1. 计算高低阈值
    ```python
    lower_boundary = img_tidu.mean() * 0.5  # 计算低阈值
    high_boundary = lower_boundary * 3  # 计算高阈值，设置为低阈值的三倍
    ```
    这里，算法计算了两个阈值：低阈值和高阈值。低阈值通常是梯度幅值图像的平均值的一半，而高阈值是低阈值的三倍。这些阈值用于确定像素是否被认为是边缘。
    
     2. 双阈值检测
    ```python
    for i in range(1, img_yizhi.shape[0] - 1):
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:
                img_yizhi[i, j] = 255  # 将该点标记为边缘点
                zhan.append([i, j])  # 将该点坐标压入栈
            elif img_yizhi[i, j] <= lower_boundary:
                img_yizhi[i, j] = 0  # 将该点标记为非边缘点
    ```
    在这个循环中，算法遍历非极大值抑制后的图像 `img_yizhi`。如果一个像素的梯度幅值大于或等于高阈值，它被立即标记为边缘点（255）。如果梯度幅值小于或等于低阈值，它被标记为非边缘点（0）。
    注意，这里不包括图像的边缘像素，因为它们的邻域不完整。
    
     3. 边缘连接
    ```python
    while not len(zhan) == 0:
        # ... （省略了部分代码）
    ```
    这部分代码使用一个栈来跟踪可能的边缘点（即那些梯度幅值在高低阈值之间的点）。算法从栈中弹出一个点，检查其8个邻域点。
    如果邻域点的梯度幅值大于低阈值且小于高阈值，该点被标记为边缘点（255），并将其坐标压入栈中。这个过程继续进行，直到栈为空。
    
     4. 清理
    ```python
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
    ```
    最后，算法清理 `img_yizhi` 中的值，确保像素点要么是0（非边缘），要么是255（边缘）。任何其他值都被设置为0。
    
    通过这一系列步骤，Canny算法能够在图像中检测出清晰的边缘。双阈值检测确保了强边缘被标记出来，而边缘连接则通过跟踪可能的边缘点来连接断开的边缘，从而形成连续的边缘线条。
    '''
    # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    # 计算低阈值，取梯度幅值图像的均值的一半
    lower_boundary = img_tidu.mean() * 0.5  # lower_boundary：低阈值

    # 计算高阈值，取低阈值的三倍
    high_boundary = lower_boundary * 3  # high_boundary：高阈值

    # 创建一个空栈，用于存储边缘点坐标
    zhan = []  # zhan：存储边缘点的栈
    # 双阈值检测
    # 遍历非极大值抑制后的图像  外圈不考虑了
    for i in range(1, img_yizhi.shape[0] - 1):  # i：行索引，从1到img_yizhi.shape[0]-2，避免边缘
        for j in range(1, img_yizhi.shape[1] - 1):  # j：列索引，从1到img_yizhi.shape[1]-2，避免边缘
            if img_yizhi[i, j] >= high_boundary:  # 如果像素值大于等于高阈值，则一定是边缘点
                img_yizhi[i, j] = 255  # 将该点标记为边缘点（255表示边缘点）
                zhan.append([i, j])  # 将该点坐标压入栈
            elif img_yizhi[i, j] <= lower_boundary:  # 如果像素值小于等于低阈值，则一定不是边缘点
                img_yizhi[i, j] = 0  # 将该点标记为非边缘点（0表示非边缘点）

    # 边缘连接
    while not len(zhan) == 0:  # 如果栈不为空，则继续处理
        temp_1, temp_2 = zhan.pop()  # 从栈中弹出一个坐标

        # 获取该点的8邻域矩阵
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]  # a：8邻域矩阵

        # 检查8邻域内的点，如果在高低阈值之间，则标记为边缘点并压入栈
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):  # 如果a[0, 0]在阈值之间
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 标记为边缘点
            zhan.append([temp_1 - 1, temp_2 - 1])  # 压入栈

        # 以下代码类似，对该点的8邻域进行检查，如果有点的值在高低阈值之间，则标记为边缘点并压入栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):  # 如果a[0, 1]在阈值之间
            img_yizhi[temp_1 - 1, temp_2] = 255  # 标记为边缘点
            zhan.append([temp_1 - 1, temp_2])  # 压入栈

        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):  # 如果a[0, 2]在阈值之间
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255  # 标记为边缘点
            zhan.append([temp_1 - 1, temp_2 + 1])  # 压入栈

        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):  # 如果a[1, 0]在阈值之间
            img_yizhi[temp_1, temp_2 - 1] = 255  # 标记为边缘点
            zhan.append([temp_1, temp_2 - 1])  # 压入栈

        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):  # 如果a[1, 2]在阈值之间
            img_yizhi[temp_1, temp_2 + 1] = 255  # 标记为边缘点
            zhan.append([temp_1, temp_2 + 1])  # 压入栈

        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):  # 如果a[2, 0]在阈值之间
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255  # 标记为边缘点
            zhan.append([temp_1 + 1, temp_2 - 1])  # 压入栈

        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):  # 如果a[2, 1]在阈值之间
            img_yizhi[temp_1 + 1, temp_2] = 255  # 标记为边缘点
            zhan.append([temp_1 + 1, temp_2])  # 压入栈

        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):  # 如果a[2, 2]在阈值之间
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255  # 标记为边缘点
            zhan.append([temp_1 + 1, temp_2 + 1])  # 压入栈

    # 清洗
    # 对双阈值检测后的图像进行处理
    for i in range(img_yizhi.shape[0]):  # i：行索引，从0到img_yizhi.shape[0]-1
        for j in range(img_yizhi.shape[1]):  # j：列索引，从0到img_yizhi.shape[1]-1
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:  # 如果像素值不是0或255
                img_yizhi[i, j] = 0  # 将其设置为0（非边缘点）

    print("img_yizhi:\n",img_yizhi,img_yizhi.shape)
    '''
    ### 代码逻辑和作用
    1. **双阈值检测**：
        - 计算低阈值和高阈值。
        - 遍历非极大值抑制后的图像，标记高于高阈值的点为边缘点（255），低于低阈值的点为非边缘点（0），并将高阈值以上的点坐标压入栈。
    2. **边缘连接**：
        - 使用栈处理边缘连接，通过检查边缘点的8邻域，如果邻域内的点值在高阈值和低阈值之间，则标记为边缘点并压入栈，确保边缘的连续性。
    3. **清洗**：
        - 遍历双阈值检测后的图像，将所有不为0和255的点设置为0，确保图像中只有边缘点（255）和非边缘点（0）。
    **总结**：这段代码实现了 Canny 边缘检测中的双阈值检测和边缘连接步骤，通过对梯度幅值图像进行处理，提取并连接图像中的边缘点，最终生成一个仅包含边缘点的二值图像。
    '''

    # 绘图
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')  # 显示最终的边缘检测结果
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()  # 显示所有图像

'''
这段代码实现了一个基于Canny算子的边缘检测算法。主要步骤如下：
读取图像并进行灰度化处理。
对灰度图像进行高斯平滑，减少噪声影响。
计算图像的梯度幅值和梯度方向。
对梯度幅值图像进行非极大值抑制，保留边缘点。
使用双阈值算法检测和连接边缘。
显示最终的边缘检测结果。
代码中使用了numpy进行数值计算，matplotlib进行图像显示。整个过程分为多个步骤，每个步骤都有详细的注释说明。


当使用费曼学习法解释Canny算法时，我们可以用简单的语言来描述它是什么以及它是如何应用的。

Canny算法是一种用于边缘检测的经典计算机视觉算法。边缘检测是图像处理中的一项重要任务，它旨在找到图像中不同区域之间的边界。Canny算法通过多个步骤来实现这一目标，以下是其中的主要步骤：

1. **降噪**：首先，对输入图像进行平滑处理，以减少噪声的影响。这通常通过应用高斯滤波器来实现，它可以模糊图像并去除高频噪声。

2. **计算梯度**：然后，计算图像中每个像素的梯度值和方向。这可以帮助我们找到图像中的边缘。

3. **非极大值抑制**：接下来，对梯度幅值进行非极大值抑制。这一步骤有助于细化边缘，只保留具有最大梯度值的像素，从而得到更细的边缘。

4. **双阈值处理**：然后，使用双阈值处理来确定真正的边缘。根据两个阈值（高阈值和低阈值），像素被分为强边缘、弱边缘或非边缘。只有强边缘被认为是最终的边缘，而弱边缘只有在与强边缘相连时才被视为边缘。

5. **边缘连接**：最后，通过连接弱边缘与强边缘的方式来完善边缘。如果弱边缘与强边缘相连，则将其视为边缘的一部分。

通过这些步骤，Canny算法能够提取出图像中的边缘，产生清晰、准确的边缘检测结果。

下面是一个简单的示例来说明Canny算法的应用。假设我们有一张黑白图像，其中包含一个圆形物体。我们希望使用Canny算法来检测出该圆形物体的边缘。

1. 首先，对图像进行降噪处理，以减少噪声的影响。

2. 然后，计算图像中每个像素的梯度值和方向。在圆形物体的边缘周围，梯度值会显著增加。

3. 接下来，对梯度幅值进行非极大值抑制，以细化边缘。

4. 然后，通过双阈值处理，将强边缘和弱边缘分开。在圆形物体的边缘处，可能会有一些强边缘。

5. 最后，通过边缘连接，将与强边缘相连的弱边缘连接起来，形成完整的圆形边缘。

这样，我们就能够使用Canny算法检测出图像中圆形物体的边缘。

通过费曼学习法，我们用简单的语言解释了Canny算法的原理和应用，并提供了一个示例来说明它如何应用于边缘检测。希望这样的解释能帮助你更好地理解Canny算法。如有任何进一步的问题，请随时提问。

'''

img = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])

print(img.mean(axis=0))
'''
每个数组的第一个列表第一个索引求平均
计算沿着第二个轴（轴索引为0）的平均值
img.mean(axis=0)：计算沿着第一个轴（轴索引为0）的平均值。即对于每列的三个像素进行平均。
对于第一列 [[1, 4, 7], [10, 13, 16], [19, 22, 25]]，我们计算其平均值：(1 + 10 + 19) / 3 = 10。
对于第二列 [[2, 5, 8], [11, 14, 17], [20, 23, 26]]，我们计算其平均值：(2 + 11 + 20) / 3 = 11。
'''

print(img.mean(axis=1))
'''
img.mean(axis=1)：计算沿着第二个轴（轴索引为1）的平均值。即对于每行的三个像素进行平均
每个数组第一列计算  
对于第一列 [1, 4, 7]，我们计算其平均值：(1 + 4 + 7) / 3 = 4。
对于第二列 [2, 5, 8]，我们计算其平均值：(2 + 5 + 8) / 3 = 5。
'''

print(img.mean(axis=2))

print(img.mean(axis=-1))
'''
即对于每个像素的三个通道进行平均,axis=-1 相当于索引到最后一位来计算
img.mean(axis=-1)：计算沿着最后一个轴（轴索引为2）的平均值。即对于每个像素的三个通道进行平均。
对于位置 (0, 0) 的像素 [1, 2, 3]，我们计算其通道的平均值：(1 + 2 + 3) / 3 = 2。
对于位置 (0, 1) 的像素 [4, 5, 6]，我们计算其通道的平均值：(4 + 5 + 6) / 3 = 5。
以此类推，计算其他位置的像素通道的平均值。
'''

print(img.mean())  # 计算整个矩阵的全局平均值


