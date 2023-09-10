import numpy as np
import matplotlib.pyplot as plt
import operator


def createDataset():
    # 四组二维特征
    X = np.array([[5, 115], [7, 106], [56, 11], [66, 9]])
    # 四组对应标签
    y = ('动作片', '动作片', '爱情片', '爱情片')
    return X, y


def classify(intX, dataSet, labels, k):
    """
     KNN算法
    """
    data_set_size = dataSet.shape[0]
    diff_mat = np.tile(intX, (data_set_size, 1)) - dataSet
    sqrt_diff_max = diff_mat ** 2

    # 计算距离
    seq_distances = sqrt_diff_max.sum(axis=1)
    distances = seq_distances ** 0.5

    # 返回distance中元素从小到大排序后的索引
    sort_distance = distances.argsort()

    class_count = {}
    for i in range(k):
        # 取出前k个元素的类别
        vote_label = labels[sort_distance[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    # reverse降序排序字典
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 结果sortedClassCount = [('动作片', 2), ('爱情片', 1)]
    return sorted_class_count[0][0]


def KNN_main(input_x: list, dataset: list, labels: tuple, k: int):
    # 计算距离
    ab_sub = np.tile(input_x, (len(dataset), 1)) - np.array(dataset)
    sqrt_sub = ab_sub ** 2
    sqrt_sum = np.sum(sqrt_sub, axis=1)
    distance = sqrt_sum ** 0.5

    sort_ids = np.argsort(distance)

    topk = sort_ids[:k]

    counter = {}
    for item in topk: counter[labels[item]] = counter.get(labels[item], 0) + 1

    sorted_counter = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_counter[0][0]


def plot_scatter(x_test, x_train, y_train):
    i, color_dict, color_list = 0, dict(), list()
    for y_hig in y_train:
        if y_hig not in color_dict:
            color_dict[y_hig] = i
            color_list.append(i)
            i += 1
        else:
            color_list.append(color_dict[y_hig])

    plt.scatter(x_train[:, 0], x_train[:, 1], c=color_list, marker="*", s=100)
    plt.scatter(x_test[0], x_test[1], c="b", marker="s")
    plt.show()


if __name__ == '__main__':
    group, labels = createDataset()
    test = [20, 101]
    test_class = KNN_main(test, group, labels, 3)
    plot_scatter(test, group, labels)
    print(test_class)
