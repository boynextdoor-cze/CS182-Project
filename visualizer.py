import os
from matplotlib import pyplot as plt
import pickle

DIRNAME = os.path.dirname(__file__)
DATAPATH = os.path.join(DIRNAME, 'data')


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def CIFAR(dataset):
    if dataset=='cifar-10':
        data = unpickle(os.path.join(DATAPATH, dataset, 'data_batch_1'))
        labels = data[b'labels']
    elif dataset =='cifar-100':
        data = unpickle(os.path.join(DATAPATH, dataset, 'train'))
        labels = data[b'fine_labels']
    label_set = set(labels)
    label_count = {}
    for label in label_set:
        label_count[label] = labels.count(label)
    plt.bar(label_count.keys(), label_count.values())
    plt.show()

if __name__ == '__main__':
    CIFAR('cifar-10')
