import random
import matplotlib.pyplot as plt


def classify_index(centroids, data_item):
    i = 0
    min_idx = 0
    min_abs = abs(centroids[0] - data_item)
    for centroid in centroids:
        diff = abs(centroid - data_item)
        if diff < min_abs:
            min_abs = diff
            min_idx = i
        i += 1
    return min_idx


def classify(centroids, classed_data):
    print("centroids", centroids)
    classify_result = [[] for _ in centroids]
    for sub_data in classed_data:
        for data_item in sub_data:
            min_idx = classify_index(centroids, data_item)
            classify_result[min_idx].append(data_item)

    return classify_result


def kmeans(origin_data, k):
    # fixme 这一组数据有时候计算出来结果方差比较大，是否有办法优化呢？
    # 感觉kmeans方法的结果具有不确定性，不一定是最优解
    print('Original data: ', origin_data)
    classed_data = [origin_data]
    for _ in range(1, k):
        classed_data.append([])

    # 不确定性应该是因为第一次随机给出质心的原因
    finally_result = classify(random.sample(origin_data, k), classed_data)
    # 如果第一次自己能给出一个合理的质心，哪怕比较靠近边缘  都可以得到一个确定的结果
    # previous_result = kmeans([20, 60, 80], [origin_data, [], []])
    while True:
        next_result = classify([sum(result_item) / len(result_item) for result_item in finally_result], finally_result)
        if finally_result == next_result:
            print('Finally result: ', next_result)
            break
        finally_result = next_result

    return finally_result


if __name__ == '__main__':
    # 准备数据
    data = random.sample(range(0, 20), 3)
    data.extend(random.sample(range(40, 60), 6))
    data.extend(random.sample(range(80, 100), 18))
    random.shuffle(data)

    # 画到第一幅图上
    plt.bar(range(len(data)), data)
    plt.title('Original')
    plt.xlabel('index')
    plt.ylabel('value')

    # kmeans分类
    result = kmeans(data, 3)

    # 将结果画到第二幅图上
    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    colors = ['blue', 'green', 'red']
    idx = 0
    for item in result:
        plt.bar([i + idx * bar_width for i in range(len(item))], item, bar_width, label=f'List {idx}',
                color=colors[idx])
        idx += 1

    plt.title('Comparison')
    plt.xlabel('index')
    plt.ylabel('value')

    plt.legend()
    plt.show()
