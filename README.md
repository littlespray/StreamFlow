# [NeurIPS' 24] StreamFlow: Streamlined Multi-Frame Optical Flow Estimation for Video Sequences

[Paper](https://arxiv.org/abs/2311.17099)

## TODO

- [ ] Clean Code
- [ ] Pip Package Support
- [x] Training & Inference Code
- [x] Pre-trained Weights




## Environment Setup

```
sh install.sh
```

([Flash-attention](https://github.com/Dao-AILab/flash-attention) is optional, supporting faster inference and less GPU memory. The implementation of StreamFlow with flash-attention support is in ``test_memory.py``.)



## Checkpoints Preparation
Download [twins_svt_large-90f6aaa9.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_svt_large-90f6aaa9.pth) and put it into the ``pretrained`` dir. This checkpoint is used in training.


Checkpoints on FlyingThings, Sintel, KITTI, and Spring could be downloaded from [here](https://drive.google.com/drive/folders/1hkSsoDGB5b59lgcZPpERUqgCTV0o82hf?usp=sharing).

## Data Preparation

```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
    ├── Spring
        ├── train
```



## Training
```
# Stage 1. On Things
sh scripts/train_things.sh 

# Stage 2. On Sintel / KITTI
sh scripts/train_sintel_kitti.sh

# Stage 3. On Spring
sh scripts/train_spring.sh
```

## Inference
```
sh scripts/infer.sh
```

## Acknowledgement
Parts of code are adapted from the following repositories. We thank the authors for their great contribution to the community:
- [RAFT](https://github.com/princeton-vl/RAFT/tree/master)
- [SKFlow](https://github.com/littlespray/SKFlow)
- [MemFlow](https://github.com/DQiaole/MemFlow)