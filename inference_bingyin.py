import cv2
import os
import logging
import argparse
import pickle

from datetime import datetime

import numpy as np
from omegaconf import OmegaConf
from typing import Dict, Tuple, Union, Optional

import torch
import decord

from decord import VideoReader
from torchvision.transforms import transforms

from transformers import AutoModel

from diffusers import AutoencoderKL, DDIMScheduler, AutoencoderKLTemporalDecoder
from diffusers.utils import check_min_version

from src.data.dwpose_utils.draw_pose import draw_pose
from src.models.rd_unet import RealisDanceUnet
from src.pipelines.pipeline import RealisDancePipeline
from src.utils.util import save_videos_grid


decord.bridge.set_bridge('torch')


def augmentation(frame, transform, state=None):
    if state is not None:
        torch.set_rng_state(state)
    return transform(frame)


def simple_reader(ref_image_path, dwpose_path, hamer_path, smpl_path, sample_size, clip_size, max_length):
    scale = (1.0, 1.0)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        # ratio is w/h
        transforms.RandomResizedCrop(
            sample_size, scale=scale,
            ratio=(sample_size[1] / sample_size[0], sample_size[1] / sample_size[0]), antialias=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    clip_transform = transforms.Compose([
        transforms.ToTensor(),
        # ratio is w/h
        transforms.RandomResizedCrop(
            clip_size, scale=scale,
            ratio=(clip_size[1] / clip_size[0], clip_size[1] / clip_size[0]), antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406],  # used for dino
                             [0.229, 0.224, 0.225],  # used for dino
                             inplace=True),
    ])
    pose_transform = transforms.Compose([
        # ratio is w/h
        transforms.RandomResizedCrop(
            sample_size, scale=scale,
            ratio=(sample_size[1] / sample_size[0], sample_size[1] / sample_size[0]), antialias=True),
    ])

    hamer_reader = VideoReader(hamer_path)
    smpl_reader = VideoReader(smpl_path)
    with open(dwpose_path, 'rb') as pose_file:
        pose_list = pickle.load(pose_file)
    
    # Get the minimum length among all inputs to ensure compatibility
    hamer_length = len(hamer_reader)
    smpl_length = len(smpl_reader)
    pose_length = len(pose_list)
    
    logging.info(f"Hamer video length: {hamer_length}")
    logging.info(f"SMPL video length: {smpl_length}")
    logging.info(f"Pose list length: {pose_length}")
    
    video_length = min(hamer_length, smpl_length, pose_length)
    logging.info(f"Using minimum length: {video_length}")
    
    # Ensure we don't exceed the available frames
    max_frames = min(video_length, max_length)
    
    # Calculate the step size to get exactly max_length frames
    if max_frames >= max_length:
        step_size = max_frames // max_length
        batch_index = list(range(0, max_frames, step_size))[:max_length]
    else:
        # If we don't have enough frames, use all available frames
        batch_index = list(range(0, max_frames))
    
    # Ensure all indices are within bounds
    batch_index = [idx for idx in batch_index if idx < len(pose_list)]
    
    # Debug information about frame sampling
    logging.info(f"max_frames: {max_frames}")
    logging.info(f"Initial batch_index length: {len(list(range(0, max_frames, 4))[:max_length])}")
    logging.info(f"Final batch_index length: {len(batch_index)}")
    logging.info(f"batch_index: {batch_index[:10]}...")  # Show first 10 indices
    
    if len(batch_index) == 0:
        raise ValueError(f"No valid frames found. video_length={video_length}, max_length={max_length}, pose_list_length={len(pose_list)}")

    hamer = hamer_reader.get_batch(batch_index).permute(0, 3, 1, 2).contiguous() / 255.0
    smpl = smpl_reader.get_batch(batch_index).permute(0, 3, 1, 2).contiguous() / 255.0

    pose = [draw_pose(pose_list[idx], hamer.shape[-2], hamer.shape[-1], draw_face=False)
            for idx in batch_index]
    pose = torch.from_numpy(
        np.stack(pose, axis=0)).permute(0, 3, 1, 2).contiguous() / 255.0

    _ref_img = cv2.cvtColor(cv2.imread(ref_image_path), cv2.COLOR_BGR2RGB)
    state = torch.get_rng_state()
    ref_image = augmentation(_ref_img, img_transform, state)
    ref_image_clip = augmentation(_ref_img, clip_transform, state)
    pose = augmentation(pose, pose_transform, state)
    hamer = augmentation(hamer, pose_transform, state)
    smpl = augmentation(smpl, pose_transform, state)

    del hamer_reader
    del smpl_reader
    return (
        ref_image.unsqueeze(0),
        ref_image_clip.unsqueeze(0),
        pose.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
        hamer.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
        smpl.permute(1, 0, 2, 3).unsqueeze(0).contiguous(),
    )


def main(
    output_dir: str,
    pretrained_model_path: str,
    pretrained_clip_path: str,
    ref_image_path: str,
    hamer_path: str,
    dwpose_path: str,
    smpl_path: str,
    sample_size: Tuple,
    clip_size: Tuple,
    max_length: int,
    fps: int,
    unet_checkpoint_path: str,
    validation_kwargs: Optional[Dict] = None,       
    save_frame: bool = False,
    train_cfg: bool = True,

    pretrained_vae_path: str = "",
    unet_additional_kwargs: Optional[Dict] = None,
    noise_scheduler_kwargs: Optional[Dict] = None,
    pose_guider_kwargs: Optional[Dict] = None,
    fusion_blocks: str = "full",
    clip_projector_kwargs: Optional[Dict] = None,
    fix_ref_t: bool = False,
    zero_snr: bool = False,
    v_pred: bool = False,
    vae_slicing: bool = False,

    mixed_precision: str = "fp16",

    global_seed: Union[int, str] = 42,
    is_debug: bool = False,
    *args,
    **kwargs,
):
    ref_name = os.path.splitext(os.path.basename(ref_image_path))[0]
    dwpose_name = os.path.splitext(os.path.basename(dwpose_path))[0]
    hamer_name = os.path.splitext(os.path.basename(hamer_path))[0]
    smpl_name = os.path.splitext(os.path.basename(smpl_path))[0]
    output_name = f"r_{ref_name}_d_{dwpose_name}_h_{hamer_name}_s_{smpl_name}"

    # check version
    check_min_version("0.30.0.dev0")

    if global_seed == "random":
        global_seed = int(datetime.now().timestamp()) % 65535

    seed = global_seed
    torch.manual_seed(seed)

    # Logging folder
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Handle the output folder creation
    os.makedirs(os.path.join(
        output_dir, 'vis', 'mp4'), exist_ok=True)
    os.makedirs(os.path.join(
        output_dir, 'vis', 'gif'), exist_ok=True)
    os.makedirs(os.path.join(
        output_dir, 'samples', 'mp4'), exist_ok=True)
    os.makedirs(os.path.join(
        output_dir, 'samples', 'gif'), exist_ok=True)

    # Load scheduler, tokenizer and models
    logging.info("Load scheduler, tokenizer and models.")
    if pretrained_vae_path != "":
        if 'SVD' in pretrained_vae_path:
            vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_vae_path, subfolder="vae")
        else:
            vae = AutoencoderKL.from_pretrained(pretrained_vae_path, subfolder="sd-vae-ft-mse")
    else:
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")

    image_encoder = AutoModel.from_pretrained(pretrained_clip_path)

    noise_scheduler_kwargs_dict = OmegaConf.to_container(
        noise_scheduler_kwargs
    ) if noise_scheduler_kwargs is not None else {}
    if zero_snr:
        logging.info("Enable Zero-SNR")
        noise_scheduler_kwargs_dict["rescale_betas_zero_snr"] = True
        if v_pred:
            noise_scheduler_kwargs_dict["prediction_type"] = "v_prediction"
            noise_scheduler_kwargs_dict["timestep_spacing"] = "linspace"
    noise_scheduler = DDIMScheduler.from_pretrained(
        pretrained_model_path,
        subfolder="scheduler",
        **noise_scheduler_kwargs_dict,
    )

    unet = RealisDanceUnet(
        pretrained_model_path=pretrained_model_path,
        image_finetune=False,
        unet_additional_kwargs=unet_additional_kwargs,
        pose_guider_kwargs=pose_guider_kwargs,
        clip_projector_kwargs=clip_projector_kwargs,
        fix_ref_t=fix_ref_t,
        fusion_blocks=fusion_blocks,
    )

    # Load pretrained unet weights
    logging.info(f"from checkpoint: {unet_checkpoint_path}")
    checkpoint = torch.load(unet_checkpoint_path, map_location="cpu", weights_only=False)
    if "global_step" in checkpoint:
        logging.info(f"global_step: {checkpoint['global_step']}")
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[7:]
        else:
            new_k = k
        new_state_dict[new_k] = state_dict[k]
    m, u = unet.load_state_dict(new_state_dict, strict=False)
    logging.info(f"Load from checkpoint with missing keys:\n{m}")
    logging.info(f"Load from checkpoint with unexpected keys:\n{u}")

    # Freeze vae and image_encoder
    vae.eval()
    vae.requires_grad_(False)
    image_encoder.eval()
    image_encoder.requires_grad_(False)
    unet.eval()
    unet.requires_grad_(False)

    # define negative prompt
    negative_prompt = "细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走"

    # Set validation pipeline
    validation_pipeline = RealisDancePipeline(
        unet=unet, vae=vae, image_encoder=image_encoder, scheduler=noise_scheduler)
    # Note: image_finetune is set during pipeline initialization
    validation_kwargs_container = {} if validation_kwargs is None else OmegaConf.to_container(validation_kwargs)
    if vae_slicing and 'SVD' not in pretrained_vae_path:
        validation_pipeline.enable_vae_slicing()

    # move to cuda
    vae.to("cuda")
    image_encoder.to("cuda")
    unet.to("cuda")
    validation_pipeline = validation_pipeline.to("cuda")

    val_ref_image, val_ref_image_clip, val_pose, val_hamer, val_smpl = simple_reader(
        ref_image_path=ref_image_path,
        dwpose_path=dwpose_path,
        hamer_path=hamer_path,
        smpl_path=smpl_path,
        sample_size=sample_size,
        clip_size=clip_size,
        max_length=max_length,
    )

    logging.info("***** Running validation *****")

    generator = torch.Generator(device=unet.device)
    generator.manual_seed(global_seed)

    height, width = sample_size

    val_ref_image = val_ref_image.to("cuda")
    val_ref_image_clip = val_ref_image_clip.to("cuda")
    val_pose = val_pose.to("cuda")
    val_hamer = val_hamer.to("cuda")
    val_smpl = val_smpl.to("cuda")

    # Predict the noise residual and compute loss
    # Mixed-precision training
    if mixed_precision in ("fp16", "bf16"):
        weight_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    else:
        weight_dtype = torch.float32
    with torch.cuda.amp.autocast(
        enabled=mixed_precision in ("fp16", "bf16"),
        dtype=weight_dtype
    ):
        sample = validation_pipeline(
            pose=val_pose,
            hamer=val_hamer,
            smpl=val_smpl,
            ref_image=val_ref_image,
            ref_image_clip=val_ref_image_clip,
            height=height, width=width,
            fake_uncond=not train_cfg,
            num_frames=max_length,
            negative_prompt=negative_prompt,
            **validation_kwargs_container).videos

    video_length = sample.shape[2]
    
    # Print actual video length information
    video_duration = video_length / fps
    logging.info(f"Generated video length: {video_length} frames")
    logging.info(f"Generated video duration: {video_duration:.2f} seconds")
    logging.info(f"Requested max_length: {max_length} frames")
    logging.info(f"Requested duration: {max_length / fps:.2f} seconds")
    
    val_ref_image = val_ref_image.unsqueeze(2).repeat(1, 1, video_length, 1, 1)
    save_obj = torch.cat([
        (val_ref_image.cpu() / 2 + 0.5).clamp(0, 1),
        val_pose.cpu(),
        val_hamer.cpu(),
        val_smpl.cpu(),
        sample.cpu(),
    ], dim=-1)

    save_path = f"{output_dir}/vis/mp4/{output_name}.mp4"
    save_videos_grid(save_obj, save_path, fps=fps)
    save_path = f"{output_dir}/vis/gif/{output_name}.gif"
    save_videos_grid(save_obj, save_path, fps=fps)
    sample_save_path = f"{output_dir}/samples/mp4/{output_name}.mp4"
    save_videos_grid(sample.cpu(), sample_save_path, fps=fps)
    sample_save_path = f"{output_dir}/samples/gif/{output_name}.gif"
    save_videos_grid(sample.cpu(), sample_save_path, fps=fps, save_frame=save_frame)
    
    # Save pose video separately
    pose_save_path = f"{output_dir}/samples/mp4/{output_name}_pose.mp4"
    save_videos_grid(val_pose.cpu(), pose_save_path, fps=fps)
    pose_save_path = f"{output_dir}/samples/gif/{output_name}_pose.gif"
    save_videos_grid(val_pose.cpu(), pose_save_path, fps=fps, save_frame=save_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--smpl", type=str, required=True)
    parser.add_argument("--hamer", type=str, required=True)
    parser.add_argument("--dwpose", type=str, required=True)
    parser.add_argument("--H", type=int, default=768)
    parser.add_argument("--W", type=int, default=576)
    parser.add_argument("--cH", type=int, default=320)
    parser.add_argument("--cW", type=int, default=240)
    parser.add_argument("--max-L", type=int, default=80)
    parser.add_argument("--fps", type=int, default=29)
    args = parser.parse_args()

    exp_config = OmegaConf.load(args.config)
    exp_config["output_dir"] = args.output
    exp_config["unet_checkpoint_path"] = args.ckpt
    exp_config["ref_image_path"] = args.ref
    exp_config["smpl_path"] = args.smpl
    exp_config["hamer_path"] = args.hamer
    exp_config["dwpose_path"] = args.dwpose
    exp_config["sample_size"] = (args.H, args.W)
    exp_config["clip_size"] = (args.cH, args.cW)
    exp_config["max_length"] = args.max_L
    exp_config["fps"] = args.fps
    main(**exp_config)
