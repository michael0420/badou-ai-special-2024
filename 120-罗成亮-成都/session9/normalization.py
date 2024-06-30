import random


def do_normalization(arr):
    arr_min = min(arr)
    arr_max = max(arr)
    return [float((x - arr_min) / (arr_max - arr_min)) for x in arr]


if __name__ == '__main__':
    x = [random.randint(0, 20) for _ in range(60)]
    print(x)
    y = do_normalization(x)
    print(y)
