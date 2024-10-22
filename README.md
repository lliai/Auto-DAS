# Auto-DAS: Automated Proxy Discovery for Training-free Distillation-aware Architecture Search

## Introduction

This is the official implementation of the paper "Auto-DAS: Automated Proxy Discovery for Training-free Distillation-aware Architecture Search" 

## Requirements
- Python 3.6
- PyTorch 1.4.0
- torchvision 0.5.0
- CUDA 10.1
- apex
- tensorboardX
- tqdm
- numpy
- scipy
- scikit-learn
- matplotlib

Optional:

- NAS-Bench-101: Subset of the dataset with only models trained at 108 epochs:

[nasbench_only108.tfrecord](https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord)

- NAS-Bench-201:

[NAS-Bench-201-v1_0-e61699.pth](https://drive.google.com/open?id=1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs)


## Usage

### 1. Prepare the dataset

Download the ImageNet dataset and put it in the folder `./data/imagenet/`. The folder structure should be like this:

```
data
  - imagenet
    - train
      - n01440764
      - n01443537
      - n01484850
      - ...
    - val
      - n01440764
      - n01443537
      - n01484850
      - ...
```

### 2. Train the teacher model

Train the teacher model by running the following command:

```
python exps/train_teacher.py --data_path ./data/imagenet/ --save_path ./exps/teacher/ --arch resnet50 --epochs 90 --batch_size 256 --lr 0.1 --lr_schedule cosine --weight_decay 1e-4 --warmup_epochs 5 --label_smoothing 0.1 --mixup_alpha 0.2 --cutout_size 16 --cutout_prob 1.0 --num_workers 8 --gpu 0
```

### 3. Train the student model

Train the student model by running the following command:

```
python exps/train_student.py --data_path ./data/imagenet/ --save_path ./exps/student/ --teacher_path ./exps/teacher/ --arch resnet50 --epochs 90 --batch_size 256 --lr 0.1 --lr_schedule cosine --weight_decay 1e-4 --warmup_epochs 5 --label_smoothing 0.1 --mixup_alpha 0.2 --cutout_size 16 --cutout_prob 1.0 --num_workers 8 --gpu 0
```

### 4. Evaluate the student model

Evaluate the student model by running the following command:

```
python exps/eval_student.py --data_path ./data/imagenet/ --save_path ./exps/student/ --teacher_path ./exps/teacher/ --arch resnet50 --batch_size 256 --num_workers 8 --gpu 0
```

## 5. Citation

If you find Auto-DAS useful in your research, please consider citing the following paper:

```
@inproceedings{sunauto,
  title={Auto-DAS: Automated Proxy Discovery for Training-free Distillation-aware Architecture Search},
  author={Sun, Haosen and Li, Lujun and Dong, Peijie and Wei, Zimian and Shao, Shitong}
  year={2024},
  organization={ECCV}
}
```

## 6. License

This project is licensed under the [MIT License](LICENSE).
