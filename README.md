# Introduction to Machine Learning (CS182) Final Project
This repository contains the code for the final project of Introduction to Machine Learning (CS182), Fall 2022, ShanghaiTech University.

Authors: Zeen Chi (chize@shanghaitech.edu.cn) and Zhongxiao Cong (congzhx@shanghaitech.edu.cn)

Almost all the code is based on [the research of Facebook](https://github.com/facebookresearch/classifier-balancing), but we made some modifications and simplifications.

### Requirements 

* Python 3
* [PyTorch](https://pytorch.org/) (version >= 0.4.1)
* [yaml](https://pyyaml.org/wiki/PyYAMLDocumentation)


### Dataset

Download the [ImageNet_2014](http://image-net.org/index) and place it in the `data/ImageNet_LT` directory, then change the `data_root` in `main.py` accordingly.

### Image Representation

1. Instance-balanced Sampling

```shell
python main.py --cfg ./config/ImageNet_LT/feat_uniform.yaml
```

2. Class-balanced Sampling

```shell
python main.py --cfg ./config/ImageNet_LT/feat_balance.yaml
```

3. Square-root Sampling

```shell
python main.py --cfg ./config/ImageNet_LT/feat_squareroot.yaml
```

4. Progressively-balancing Sampling

```shell
python main.py --cfg ./config/ImageNet_LT/feat_shift.yaml
```

Test the joint learned classifier with representation learning

```shell
python main.py --cfg ./config/ImageNet_LT/feat_uniform.yaml --test 
```

### Classifier Learning 

The commands below only illustrate how to execute classifier learning based on the instance-balanced sampling method. For other re-sampling methods, please change the path to the corresponding configuration. 

1. Nearest Class Mean classifier (NCM).

```shell
python main.py --cfg ./config/ImageNet_LT/feat_uniform.yaml --test --knn
```

2. Classifier Re-training (cRT)

```shell
python main.py --cfg ./config/ImageNet_LT/cls_crt.yaml --model_dir ./logs/ImageNet_LT/models/resnext50_uniform_e90
python main.py --cfg ./config/ImageNet_LT/cls_crt.yaml --test
```

3. Tau-normalization 

Extract fatures

```shell
for split in train_split val test
do
  python main.py --cfg ./config/ImageNet_LT/feat_uniform.yaml --test --save_feat $split
done
```

Evaluation

```shell
for split in train val test
do
  python tau_norm.py --root ./logs/ImageNet_LT/models/resnext50_uniform_e90/ --type $split
done
```
