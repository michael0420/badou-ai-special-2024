"""
定义其他函数或工具
"""
import numpy as np

def print_top5_predictions(y_pred, path_synset):
    with open(path_synset) as f:
        synset = [line.strip() for line in f.readlines() ]

    predictions = np.argsort(y_pred,axis=1)[:,::-1] # 按列从小到大排列，然后数据按列倒序排列
    for i, prediction in enumerate(predictions):
        print(f"img {i+1}:")
        print(f"    top1： {synset[prediction[0]], y_pred[i, prediction[0]]}")
        print(f"    top5： {[synset[prediction[index]] for index in range(5)]}")


