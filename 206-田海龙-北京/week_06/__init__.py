import inspect
import sys
import os

# 添加根目录到系统路径下，这样便可以引用根目录下模块
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from utils import say_hello, cv_imread, cv_set_titile

# 当前文件路径
current_file_path = inspect.getframeinfo(inspect.currentframe()).filename
# 当前文件所在路径，方便拼接当前路径下文件，不需要从根目录一级一级拼接
current_directory = os.path.dirname(os.path.abspath(current_file_path))


def main():
    res = say_hello("my little cute")


if __name__ == "__main__":
    main()
