from collections import defaultdict
import os
from matplotlib import pyplot as plt
import pickle, csv
import numpy as np

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

def ImageNet():
    path=os.path.join(DATAPATH,'ImageNet_LT','ImageNet_LT_train.txt')
    num_per_cls=defaultdict(lambda: 0)

    with open(path,'r') as f:
        trainData=csv.reader(f)
        for row in trainData:
            tmp=row[0].split()
            cls_id=tmp[1]
            num_per_cls[cls_id]+=1

    num_per_cls={k: v for k, v in sorted(num_per_cls.items(), key=lambda x: x[1],reverse=True)}
    cls_ids=range(len(num_per_cls.keys()))
    cls_nums=list(num_per_cls.values())
    
    plt.figure()
    plt.plot(cls_ids,cls_nums)
    plt.ylim(0,1400)
    plt.fill_between(x=cls_ids[:300],y1=0,y2=cls_nums[:300],facecolor='red', alpha=0.5)
    plt.fill_between(x=cls_ids[300:],y1=0,y2=cls_nums[300:],facecolor='green', alpha=0.5)
    plt.xlabel('Sorted Class Index')
    plt.ylabel('Number of Instances')
    plt.savefig('img/ImageNet_LT.jpg',dpi=300)

if __name__ == '__main__':
    #CIFAR('cifar-10')
    ImageNet()