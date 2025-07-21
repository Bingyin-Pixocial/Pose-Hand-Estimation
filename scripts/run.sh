# cd ../
# CUDA_VISIBLE_DEVICES=4 python inference_bingyin.py \
# --config configs/stage2_hamer.yaml \
# --smpl ../../motion_control/RealisDance-DiT/pose/smpl/1.mp4 \
# --hamer ../../motion_control/RealisDance-DiT/pose/hamer/1.mp4 \
# --dwpose ../../motion_control/RealisDance-DiT/pose/dwpose/1.pkl \
# --W 576 \
# --H 1024 \
# --max-L 151 \
# --fps 30 \
# --ckpt ./ckpts/stage_2_hamer_release.ckpt \
# --ref ../../motion_control/RealisDance-DiT/pose/ref/3.jpg \
# --output ../../motion_control/RealisDance-DiT/pose/rd_unet/3_30fps

# CUDA_VISIBLE_DEVICES=4 python inference_bingyin.py \
# --config configs/stage2_hamer.yaml \
# --smpl ../../motion_control/RealisDance-DiT/pose/smpl/1.mp4 \
# --hamer ../../motion_control/RealisDance-DiT/pose/hamer/1.mp4 \
# --dwpose ../../motion_control/RealisDance-DiT/pose/dwpose/1.pkl \
# --W 576 \
# --H 1024 \
# --max-L 151 \
# --fps 30 \
# --ckpt ./ckpts/stage_2_hamer_release.ckpt \
# --ref ../../motion_control/RealisDance-DiT/pose/ref/4.jpg \
# --output ../../motion_control/RealisDance-DiT/pose/rd_unet/4_30fps

cd ../
CUDA_VISIBLE_DEVICES=4 python inference_bingyin.py \
--config configs/stage2_hamer.yaml \
--smpl ../../motion_control/RealisDance-DiT/pose/smpl/1.mp4 \
--hamer ../../motion_control/RealisDance-DiT/pose/hamer/1.mp4 \
--dwpose ../../motion_control/RealisDance-DiT/pose/dwpose/1.pkl \
--W 576 \
--H 1024 \
--max-L 151 \
--fps 30 \
--ckpt ./ckpts/stage_2_hamer_release.ckpt \
--ref ../../motion_control/RealisDance-DiT/pose/ref/2.png \
--output ../../motion_control/RealisDance-DiT/pose/rd_unet/2_30fps

CUDA_VISIBLE_DEVICES=4 python inference_bingyin.py \
--config configs/stage2_hamer.yaml \
--smpl ../../motion_control/RealisDance-DiT/pose/smpl/1.mp4 \
--hamer ../../motion_control/RealisDance-DiT/pose/hamer/1.mp4 \
--dwpose ../../motion_control/RealisDance-DiT/pose/dwpose/1.pkl \
--W 576 \
--H 1024 \
--max-L 151 \
--fps 30 \
--ckpt ./ckpts/stage_2_hamer_release.ckpt \
--ref ../../motion_control/RealisDance-DiT/pose/ref/1.png \
--output ../../motion_control/RealisDance-DiT/pose/rd_unet/1_30fps


# CUDA_VISIBLE_DEVICES=4 python inference_bingyin.py \
# --config configs/stage2_hamer.yaml \
# --smpl ../../motion_control/RealisDance-DiT/pose/smpl/1.mp4 \
# --hamer ../../motion_control/RealisDance-DiT/pose/hamer/1.mp4 \
# --dwpose ../../motion_control/RealisDance-DiT/pose/dwpose/1.pkl \
# --max-L 151 \
# --fps 30 \
# --W 576 \
# --H 1024 \
# --ckpt ./ckpts/stage_2_hamer_release.ckpt \
# --ref ../../motion_control/UniAnimate-DiT/data/images/WOMEN-Blouses_Shirts-id_00004955-01_4_full.jpg \
# --output ../../motion_control/RealisDance-DiT/pose/rd_unet/women_30fps

# CUDA_VISIBLE_DEVICES=4 python inference_bingyin.py \
# --config configs/stage2_hamer.yaml \
# --smpl __assets__/demo_seq/smpl_1.mp4 \
# --hamer __assets__/demo_seq/hamer_1.mp4 \
# --dwpose __assets__/demo_seq/dwpose_1.pkl \
# --ckpt ./ckpts/stage_2_hamer_release.ckpt \
# --ref ../../motion_control/RealisDance-DiT/pose/ref/2.png \
# --output ../../motion_control/RealisDance-DiT/pose/output/test2


# CUDA_VISIBLE_DEVICES=4 python inference_bingyin.py \
# --config configs/stage2_hamer.yaml \
# --smpl __assets__/demo_seq/smpl_1.mp4 \
# --hamer __assets__/demo_seq/hamer_1.mp4 \
# --dwpose __assets__/demo_seq/dwpose_1.pkl \
# --ckpt ./ckpts/stage_2_hamer_release.ckpt \
# --ref ../../motion_control/RealisDance-DiT/pose/ref/1.png \
# --output ../../motion_control/RealisDance-DiT/pose/output/test1