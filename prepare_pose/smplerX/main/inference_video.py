import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
import cv2
from tqdm import tqdm
import json
from typing import Literal, Union
from mmdet.apis import init_detector, inference_detector
from utils.inference_utils import process_mmdet_results, non_max_suppression
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--exp_name', type=str, default='output/test')
    parser.add_argument('--pretrained_model', type=str, default="smpler_x_h32")
    parser.add_argument('--testset', type=str, default='EHF')
    parser.add_argument('--agora_benchmark', type=str, default='na')
    parser.add_argument('--video_path', type=str, default='input.mp4')  
    parser.add_argument('--output_path', type=str, default='out_demo') 
    parser.add_argument('--demo_dataset', type=str, default='na')
    parser.add_argument('--demo_scene', type=str, default='all')
    parser.add_argument('--show_verts', action="store_true")
    parser.add_argument('--show_bbox', action="store_true")
    parser.add_argument('--save_mesh', action="store_true")
    parser.add_argument('--multi_person', action="store_true")
    parser.add_argument('--iou_thr', type=float, default=0.5)
    parser.add_argument('--bbox_thr', type=int, default=50)
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    config_path = osp.join('./config', f'config_{args.pretrained_model}.py')
    ckpt_path = osp.join('../pretrained_models', f'{args.pretrained_model}.pth.tar')

    cfg.get_config_fromfile(config_path)
    cfg.update_test_config(args.testset, args.agora_benchmark, shapy_eval_split=None, 
                            pretrained_model_path=ckpt_path, use_cache=False)
    cfg.update_config(args.num_gpus, args.exp_name)
    cudnn.benchmark = True

    # load model
    from base import Demoer
    from utils.preprocessing import load_img, process_bbox, generate_patch_image
    from utils.vis import save_obj,render_mesh_kugang, render_mesh
    from utils.human_models import smpl_x
    demoer = Demoer()
    demoer._make_model()
    demoer.model.eval()
    
    multi_person = args.multi_person
            

    ### mmdet init
    checkpoint_file = '../pretrained_models/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    config_file= '../pretrained_models/mmdet/mmdet_faster_rcnn_r50_fpn_coco.py'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
    
    video_capture = cv2.VideoCapture(args.video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    os.makedirs(args.output_path, exist_ok=True)
    out_video_path = os.path.join(args.output_path, 'smplx_video.mp4')
    output_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_idx = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_idx += 1
        original_img = frame
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]

        ## mmdet inference
        mmdet_results = inference_detector(model, frame)
        mmdet_box = process_mmdet_results(mmdet_results, cat_id=0, multi_person=multi_person)

        print(f"Frame {frame_idx}: Detected {len(mmdet_box[0])} persons")
        
        if len(mmdet_box[0]) < 1:
            print(f"Frame {frame_idx}: No persons detected, writing black frame")
            vis_img = np.zeros_like(vis_img)  # Create black image of the same size
            output_video.write(vis_img)
            continue
        
        if not multi_person:
            num_bbox = 1
            mmdet_box = mmdet_box[0]
        else:
            mmdet_box = non_max_suppression(mmdet_box[0], args.iou_thr)
            num_bbox = len(mmdet_box)

        for bbox_id in range(num_bbox):
            print(f"Frame {frame_idx}: Processing bbox {bbox_id + 1}/{num_bbox}")
            mmdet_box_xywh = np.zeros((4))
            mmdet_box_xywh[0] = mmdet_box[bbox_id][0]
            mmdet_box_xywh[1] = mmdet_box[bbox_id][1]
            mmdet_box_xywh[2] = abs(mmdet_box[bbox_id][2] - mmdet_box[bbox_id][0])
            mmdet_box_xywh[3] = abs(mmdet_box[bbox_id][3] - mmdet_box[bbox_id][1])
            print(f"Frame {frame_idx}: Raw bbox xywh: {mmdet_box_xywh}")

            if mmdet_box_xywh[2] < args.bbox_thr or mmdet_box_xywh[3] < args.bbox_thr * 3:
                print(f"Frame {frame_idx}: Bbox too small, skipping. Width: {mmdet_box_xywh[2]}, Height: {mmdet_box_xywh[3]}, Threshold: {args.bbox_thr}")
                continue

            start_point = (int(mmdet_box[bbox_id][0]), int(mmdet_box[bbox_id][1]))
            end_point = (int(mmdet_box[bbox_id][2]), int(mmdet_box[bbox_id][3]))

            bbox = process_bbox(mmdet_box_xywh, original_img_width, original_img_height)
            print(f"Frame {frame_idx}: Processed bbox: {bbox}")
            
            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
            print(f"Frame {frame_idx}: Generated patch image shape: {img.shape}")
            
            img = transforms.ToTensor()(img.astype(np.float32)) / 255
            img = img.cuda()[None, :, :, :]
            print(f"Frame {frame_idx}: Input tensor shape: {img.shape}")
            
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')
            
            # Debug: Check if model output contains the expected key
            if 'smplx_mesh_cam' not in out:
                print(f"Frame {frame_idx}: 'smplx_mesh_cam' not found in model output. Available keys: {list(out.keys())}")
                continue
                
            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
            
            # Debug: Check mesh shape and content
            print(f"Frame {frame_idx}: mesh shape: {mesh.shape}")
            if mesh.shape[0] == 0:
                print(f"Frame {frame_idx}: Empty mesh detected!")
                continue
            print(f"Frame {frame_idx}: mesh min/max values: {mesh.min():.3f}/{mesh.max():.3f}")
            
            # Debug: Check face data
            print(f"Frame {frame_idx}: face shape: {smpl_x.face.shape}")
            if smpl_x.face.shape[0] == 0:
                print(f"Frame {frame_idx}: Empty face data detected!")
                continue

            focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
            princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
            
            # Debug: Check camera parameters
            print(f"Frame {frame_idx}: focal: {focal}, princpt: {princpt}")
            
            try:
                vis_img = render_mesh_kugang(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=args.show_verts)
                print(f"Frame {frame_idx}: render_mesh_kugang completed successfully")
            except Exception as e:
                print(f"Frame {frame_idx}: Error in render_mesh_kugang: {str(e)}")
                # Continue with original image if rendering fails
                continue

            if args.show_bbox:
                vis_img = cv2.rectangle(vis_img, start_point, end_point, (255, 0, 0), 2)
        vis_img = vis_img.astype('uint8')
        output_video.write(vis_img[:, :, ::-1])

    video_capture.release()
    output_video.release()
    print("Video processing completed.")

if __name__ == "__main__":
    main()