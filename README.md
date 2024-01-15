<img width="615" alt="image" src="https://github.com/Shruti-Bansal/Dual-Attention-Network-for-Scene-Segmentation/assets/23707426/dd5c8b25-2b34-499c-b961-796d5fe8b997">


# DANet
A PyTorch implementation of DANet based on CVPR 2019 paper [Dual Attention Network for Scene Segmentation](https://arxiv.org/abs/1809.02983). 

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
- opencv
```
pip install opencv-python
```
- tensorboard
```
pip install tensorboard
```
- pycocotools
```
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```
- fvcore
```
pip install git+https://github.com/facebookresearch/fvcore
```
- cityscapesScripts
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```
- detectron2
```
pip install git+https://github.com/facebookresearch/detectron2.git@master
```

## Datasets
For a few datasets that detectron2 natively supports, the datasets are assumed to exist in a directory called
`datasets/`, under the directory where you launch the program. They need to have the following directory structure:

### Expected dataset structure for Cityscapes:
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
```
run `./datasets/prepare_cityscapes.py` to creat `labelTrainIds.png`.

## Training
To train a model, run
```bash
python train_net.py --config-file <config.yaml>
```

For example, to launch end-to-end DANet training with ResNet-50 backbone on 8 GPUs, one should execute:
```bash
python train_net.py --config-file configs/r50.yaml --num-gpus 8
```

## Evaluation
Model evaluation can be done similarly:
```bash
python train_net.py --config-file configs/r50.yaml --num-gpus 8 --eval-only MODEL.WEIGHTS checkpoints/model.pth
```

