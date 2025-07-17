# Pose&Hand Estimation

## Prepare Pretrained Checkpoints

```
mkdir pretrained_models
cd pretrained_models

# prepare rv-5-1 from https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/tree/main
mkdir RV
cd RV
ln -s $PATH-TO-RV-5-1 rv-5-1
cd ../

# prepare dino v2 from https://huggingface.co/facebook/dinov2-large/tree/main
mkdir DINO
cd DINO
ln -s $PATH-TO-DINOV2 dinov2
cd ../

# prepare motion module from https://github.com/guoyww/AnimateDiff
mkdir MM
cd MM
download mm_v2.ckpt from [here](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt)

(Optional)
# prepare sd-vae-ft-mse from https://huggingface.co/stabilityai/sd-vae-ft-mse
# link to a SD dir with a subfolder named 'sd-vae-ft-mse' 
```

## Set up the pose and hand estimation and reconstruction repos

You can generate your `DWPose`, `HaMeR`, or `SMPLerX` data by following the instructions in the [Prepare Pose README](prepare_pose/README.md).

## Train Your RealisDance

```
# stage1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 \
    train.py --config configs/stage1_hamer.yaml 

# stage2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 \
    train.py --config configs/stage2_hamer.yaml 
```

## Evaluate Your RealisDance

```
# stage1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 \
    evaluate.py --config configs/stage1_hamer.yaml --output $PATH-TO-OUTPUT --ckpt $PATH-TO-CKPT

# stage2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 \
    evaluate.py --config configs/stage2_hamer.yaml --output $PATH-TO-OUTPUT --ckpt $PATH-TO-CKPT
```

## Disclaimer
This project is released for academic use.
We disclaim responsibility for user-generated content.

## Contact Us
Jingkai Zhou: [fs.jingkaizhou@gmail.com](mailto:fs.jingkaizhou@gmail.com)


## BibTeX
```
@article{zhou2024realisdance,
  title={RealisDance: Equip controllable character animation with realistic hands},
  author={Zhou, Jingkai and Wang, Benzhi and Chen, Weihua and Bai, Jingqi and Li, Dongyang and Zhang, Aixi and Xu, Hao and Yang, Mingyang and Wang, Fan},
  journal={arXiv preprint arXiv:2409.06202},
  year={2024}
}
```


## Acknowledgements
Codebase built upon [Open-Animate Anyone](https://github.com/guoqincode/Open-AnimateAnyone), [Moore-Animate Anyone](https://github.com/MooreThreads/Moore-AnimateAnyone), and [MusePose](https://github.com/TMElyralab/MusePose).
