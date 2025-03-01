## Revisiting Adversarial Training under Long-Tailed Distributions

Code for CVPR 2024 "[Revisiting Adversarial Training under Long-Tailed Distributions](https://openaccess.thecvf.com/content/CVPR2024/html/Yue_Revisiting_Adversarial_Training_Under_Long-Tailed_Distributions_CVPR_2024_paper.html)".

## Environment

- Python (3.9.12)
- Pytorch (2.1.0)
- torchvision (0.16.0)
- CUDA
- AutoAttack
- advertorch

## Content

- ```./datasets```: Generate long-tailed datasets.
- ```./models```: Models used for training.
- ```train_at_bsl_cifar10.py```: AT-BSL on CIFAR-10-LT.
- ```train_at_bsl_cifar100.py```: AT-BSL on CIFAR-100-LT.
- ```train_at_bsl_tiny_imagenet.py```: AT-BSL on Tiny-ImageNet-LT.
- ```at_bsl_loss.py```: Loss function for AT-BSL.
- ```pgd_attack.py```: Use PGD to select the best epoch during training.
- ```eval.py```:  Evaluate the robustness under various adversarial attacks.

## Run

- AT-BSL using ResNet18 on CIFAR-10-LT
```bash
CUDA_VISIBLE_DEVICES='0' python train_at_bsl_cifar10.py --arch res --aug none
```

- AT-BSL-RA using ResNet18 on CIFAR-10-LT
```bash
CUDA_VISIBLE_DEVICES='0' python train_at_bsl_cifar10.py --arch res --aug ra
```

- AT-BSL using WideResNet-34-10 on CIFAR-100-LT

```bash
CUDA_VISIBLE_DEVICES='0' python train_at_bsl_cifar100.py --arch wrn --aug none
```

- AT-BSL-AuA using WideResNet-34-10 on CIFAR-100-LT

```bash
CUDA_VISIBLE_DEVICES='0' python train_at_bsl_cifar100.py --arch wrn --aug aua
```

- AT-BSL-RA using PreActResNet-18 on Tiny-ImageNet-LT

```bash
CUDA_VISIBLE_DEVICES='0' python train_at_bsl_tiny_imagenet.py --aug ra
```

- Evaluation

```bash
CUDA_VISIBLE_DEVICES='0' python eval.py --model_path INSERT-YOUR-MODEL-PATH
```

## Pre-trained Models

- The pre-trained models can be downloaded from the [Google Drive](https://drive.google.com/drive/folders/1o-ZGm43jfrg_tALUNy_dGJpUBx1yHfp-?usp=drive_link).

## Reference Code

[1] RoBal: https://github.com/wutong16/Adversarial_Long-Tail

[2] REAT: https://github.com/GuanlinLee/REAT

[3] AT: https://github.com/MadryLab/cifar10_challenge

[4] TRADES: https://github.com/yaodongyu/TRADES

## Citation
```
@inproceedings{yue2024revisiting,
  title={Revisiting Adversarial Training under Long-Tailed Distributions},
  author={Yue, Xinli and Mou, Ningping and Wang, Qian and Zhao, Lingchen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24492--24501},
  year={2024}
}
```