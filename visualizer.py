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
    
    plt.figure(figsize=(6,4))
    plt.plot(cls_ids,cls_nums)
    plt.ylim(0,1400)
    plt.fill_between(x=cls_ids[:450],y1=0,y2=cls_nums[:450],facecolor='red', alpha=0.5)
    plt.fill_between(x=cls_ids[450:],y1=0,y2=cls_nums[450:],facecolor='green', alpha=0.5)
    plt.xlabel('Sorted Class Index')
    plt.ylabel('Number of Instances')
    plt.title('ImageNet_LT Category Instances Distribution')
    plt.text(150, 300, 'Head Classes', fontsize=12, font='Times New Roman')
    plt.text(600, 100, 'Tail Classes', fontsize=12, font='Times New Roman')
    plt.savefig('img/ImageNet_LT.jpg',dpi=300)

def result():
    labels = ['instance', 'class', 'square', 'progressive']
    joint = [65.9, 61.8, 64.3, 61.9]
    ncm = [56.6, 58.4, 59.0, 57.8]
    crt = [61.8, 61.3, 62.6, 61.1]
    tau = [59.1, 61.4, 62.8, 61.6]

    x = np.arange(len(labels))
    width = 0.2

    fig = plt.figure(figsize=(20,15))
    axes = fig.subplots(2,2)
    
    axes[0,0].bar(x - 1.5*width, joint, width, label='Joint')
    axes[0,0].bar(x - 0.5*width, ncm, width, label='NCM')
    axes[0,0].bar(x + 0.5*width, crt, width, label='cRT')
    axes[0,0].bar(x + 1.5*width, tau, width, label='Tau')
    axes[0,0].set_ylabel('Accuracy %', fontsize=20)
    axes[0,0].set_title('Many-Shot', fontsize=20)
    axes[0,0].set_xticks(x, labels=labels, fontsize=20)


    joint = [37.5, 40.1, 41.2, 43.2]
    ncm = [45.3, 40.1, 44.7, 43.0]
    crt = [46.2, 39.3, 44.5, 43.1]
    tau = [46.9, 40.2, 42.0, 43.3]

    axes[0,1].bar(x - 1.5*width, joint, width, label='Joint')
    axes[0,1].bar(x - 0.5*width, ncm, width, label='NCM')
    axes[0,1].bar(x + 0.5*width, crt, width, label='cRT')
    axes[0,1].bar(x + 1.5*width, tau, width, label='Tau')
    axes[0,1].set_ylabel('Accuracy %', fontsize=20)
    axes[0,1].set_title('Medium-Shot', fontsize=20)
    axes[0,1].set_xticks(x, labels=labels, fontsize=20)


    joint = [7.7, 15.5, 17.0, 19.4]
    ncm = [28.1, 18.0, 24.5, 21.6]
    crt = [27.4, 15.2, 22.0, 19.0]
    tau = [30.7, 15.3, 24.8, 19.2]

    axes[1,0].bar(x - 1.5*width, joint, width, label='Joint')
    axes[1,0].bar(x - 0.5*width, ncm, width, label='NCM')
    axes[1,0].bar(x + 0.5*width, crt, width, label='cRT')
    axes[1,0].bar(x + 1.5*width, tau, width, label='Tau')
    axes[1,0].set_ylabel('Accuracy %', fontsize=20)
    axes[1,0].set_title('Few-Shot', fontsize=20)
    axes[1,0].set_xticks(x, labels=labels, fontsize=20)
    

    joint = [44.4, 45.1, 46.8, 47.2]
    ncm = [47.3, 44.2, 47.5, 45.1]
    crt = [49.6, 44.5, 47.9, 46.7]
    tau = [49.4, 44.8, 47.6, 47.1]

    axes[1,1].bar(x - 1.5*width, joint, width, label='Joint')
    axes[1,1].bar(x - 0.5*width, ncm, width, label='NCM')
    axes[1,1].bar(x + 0.5*width, crt, width, label='cRT')
    axes[1,1].bar(x + 1.5*width, tau, width, label='Tau')
    axes[1,1].set_ylabel('Accuracy %', fontsize=20)
    axes[1,1].set_title('All', fontsize=20)
    axes[1,1].set_xticks(x, labels=labels, fontsize=20)

    h,l = axes.flatten()[-1].get_legend_handles_labels()
    handles = h
    labels = l
    fig.legend(handles, labels, loc = 'upper center', fontsize=20, ncol=4)
    plt.savefig('img/result.jpg',dpi=300)
    plt.show()

if __name__ == '__main__':
    #ImageNet()
    result()