from modules.global_params import PATH_TRIAN_DATASHEET,PATH_TEST_DATASHEET,PATH_TRIAN_IMGS_DIR,PATH_TEST_IMGS_DIR
import os

def _generate_datasheet(imgs_dir_path, datasheet_path):
    """
    用于生成datasheet.txt保存数据条目，每行代表一条数据：
    每行格式：图片路径,标签
    """

    # 获取所有图片文件名
    imgs = os.listdir(imgs_dir_path)

    # 生成datasheet.txt文件
    with open(datasheet_path, 'w') as f:
        for img in imgs:
            img_path = os.path.join(imgs_dir_path, img)
            label_text = img.split('.')[0]
            if label_text == "dog":
                f.write(','.join([img_path, "1\n"]))
            elif label_text == "cat":
                f.write(','.join([img_path, "0\n"]))

if __name__ == "__main__":

    # 生成训练数据datasheet
    _generate_datasheet(imgs_dir_path=PATH_TRIAN_IMGS_DIR, datasheet_path=PATH_TRIAN_DATASHEET)

    # 生成测试数据datasheet
    _generate_datasheet(imgs_dir_path=PATH_TEST_IMGS_DIR, datasheet_path=PATH_TEST_DATASHEET)


