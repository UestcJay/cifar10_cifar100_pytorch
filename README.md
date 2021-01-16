# cifar10_cifar100_pytorch
Practice on cifar100(ResNet, Mobilenet,Mobilenetv2) ,there will be more.
# Requirements
This is my experiment eviroument

python3.5
pytorch1.2.0+cuda10.0
tensorboard 2.2.2(optional)
# Usage

```bash
git clone https://github.com/UestcJay/cifar10_cifar100_pytorch.git
```
## 1. enter directory

```cpp
cd cifar10_cifar100_pytorch
```
## 2. dataset
you can use cifar100 and cifar10 dataset from torchvision directly as in the code, but most time it is not convenient for us to download the datasets for chinese, then you can download from [cifar10 and cifar100](https://pan.baidu.com/s/1lyqTifTNhCvRdve4YVGJIQ), the key is **kyfq**. If the link fails, just feel free to open a issue to inform me. I also kept the sample code for writing your own dataset module in **dataset.py**. You can refer it!
## 3.run tensorbard(optional)



```bash
pip install tensorboard
mkdir runs
Run tensorboard
tensorboard --logdir='runs' --port=6006 --host='localhost'
```
## 4.train the model

```python
python train.py #train cifar100
python train_cifar10.py #train cifar10
```
You can change the default parameters to select the corresponding network.
sometimes, you might want to use `warmup` training by set -warm to 1 or 2, to prevent network diverge during early training phase.
## 5. test the model

```python
python test.py
```
it reports top-1 error and top-5 error.

If you meet any question, just feel free to open a new issue. Thanks for your attention!

# Acknowledge
I refer the [repo](https://github.com/weiaicunzai/pytorch-cifar100) for coding and this [blog](https://www.cnblogs.com/yanshw/p/12563872.html) .
