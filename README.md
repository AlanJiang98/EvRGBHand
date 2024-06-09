# EvRGBHand [CVPR'24] ✨✨

[[:page_with_curl:Project Page](https://alanjiang98.github.io/evrgbhand.github.io/)]  [[Data](https://www.dropbox.com/scl/fo/fknqrn1jb2bh9088gqw8w/AFiaujVeKt36ee2Roc8q8VE?rlkey=21atcy255w45z3du2whz4uzpb&e=2&st=dfsiib22&dl=0)] [[Paper](https://arxiv.org/abs/2403.07346)] [[Models](https://drive.google.com/file/d/19dB8KSkdk502l4hZQUY3Eo24Vcw1tHBH/view?usp=drive_link)]
[[:movie_camera:Video](https://www.youtube.com/watch?v=cEVkQ10fWh0)]

*This is the official PyTorch implementation of [Complementing Event Streams and RGB Frames for Hand Mesh Reconstruction](https://github.com/AlanJiang98/EvRGBHand).This work investigates the feasibility of using events and images for HMR, and proposes the first solution to 3D HMR by complementing event streams and RGB frames.*
<div  align="center">    
<img src="figure/teaser.png" width="600" height="300" alt="teaser" /> 
</div>

## Usage 

### Installation
```bash
# Create a new environment
conda create --name evrgb python=3.9
conda activate evrgb

# Install Pytorch
conda install pytorch=2.1.0  torchvision=0.16.0  pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Pytorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

# Install requirements
pip install -r requirements.txt
```
Our codebase is developed based on Ubuntu 23.04 and NVIDIA GPU cards. 


### Data Preparation
- Download the EvRealHands dataset from [EvHandPose](https://alanjiang98.github.io/evhandpose.github.io/) and change the path in the ``src/datasets/dataset.yaml`` .
- Download MANO models from [MANO](https://mano.is.tue.mpg.de/download.php). Put the ```MANO_LEFT.pkl``` and ```MANO_RIGHT.pkl``` to ```models/mano```.

### Train
- Modify the config file in ``src/configs/config`` .

```shell
python train.py  --config <config-path> 
```

### Evaluation

- Download the pretrained model for the [EvImHandNet](https://drive.google.com/file/d/19dB8KSkdk502l4hZQUY3Eo24Vcw1tHBH/view?usp=drive_link)
```shell
python train.py --config <config-path>  --resume_checkpoint <pretrained-model> --config_merge <eval-config-path>  --run_eval_only --output_dir <output-dir>

#For example 
python train.py --config src/configs/config/evrgbhand.yaml --resume_checkpoint output/EvImHandNet.pth --config_merge src/configs/config/eval_temporal.yaml --run_eval_only --output_dir result/evrgbhand/
```

## Citation

```bibtex
@inproceedings{Jiang2024EvRGBHand,
      title={Complementing Event Streams and RGB Frames for Hand Mesh Reconstruction}, 
      author={Jiang, Jianping and Zhou, Xinyu and Wang, Bingxuan and Deng, Xiaoming and Xu, Chao and Shi, Boxin},
      booktitle={CVPR},
      year={2024}
}

@article{jiang2024evhandpose,
  author    = {Jianping, Jiang and Jiahe, Li and Baowen, Zhang and Xiaoming, Deng and Boxin, Shi},
  title     = {EvHandPose: Event-based 3D Hand Pose Estimation with Sparse Supervision},
  journal   = {TPAMI},
  year      = {2024},
}
```

## Acknowledgement
- Our code is based on [FastMETRO](https://github.com/postech-ami/FastMETRO). 
- In our experiments, we use the official code of [MeshGraphormer](https://github.com/microsoft/MeshGraphormer), [FastMETRO](https://github.com/postech-ami/FastMETRO), [EventHands](https://github.com/r00tman/EventHands) for comparison. We sincerely recommend that you read these papers to fully understand the method behind EvRGBHand.

## Related Projects
- [EvHandPose: Event-based 3D Hand Pose Estimation with Sparse Supervision](https://alanjiang98.github.io/evhandpose.github.io/)
- [EvPlug: Learn a Plug-and-Play Module for Event and Image Fusion](https://arxiv.org/abs/2312.16933)