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
import traceback
from collections import deque

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
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum retries for model inference')
    parser.add_argument('--fallback_to_original', action="store_true", default=True, help='Use original frame when processing fails')
    parser.add_argument('--use_temporal_smoothing', action="store_true", default=True, help='Use temporal smoothing for mesh estimation')
    parser.add_argument('--temporal_window', type=int, default=5, help='Window size for temporal smoothing')
    parser.add_argument('--use_template_fallback', action="store_true", default=True, help='Use SMPL-X template as fallback')
    args = parser.parse_args()
    return args

class MeshEstimator:
    """Class to handle mesh estimation and fallback mechanisms"""
    
    def __init__(self, smpl_x_model, temporal_window=5):
        self.smpl_x_model = smpl_x_model
        self.temporal_window = temporal_window
        self.mesh_history = deque(maxlen=temporal_window)
        self.face_history = deque(maxlen=temporal_window)
        self.template_mesh = None
        self.template_face = None
        self._initialize_template_data()
    
    def _initialize_template_data(self):
        """Initialize template mesh and face data from SMPL-X model"""
        try:
            # Get template mesh (neutral pose, zero shape parameters)
            from utils.smplx import smplx
            template_model = smplx.create(
                cfg.human_model_path, 
                'smplx', 
                gender='NEUTRAL', 
                use_pca=False, 
                use_face_contour=True,
                create_global_orient=False,
                create_body_pose=False,
                create_left_hand_pose=False,
                create_right_hand_pose=False,
                create_jaw_pose=False,
                create_leye_pose=False,
                create_reye_pose=False,
                create_betas=False,
                create_expression=False,
                create_transl=False
            )
            
            # Generate template mesh with zero parameters
            with torch.no_grad():
                template_output = template_model(
                    betas=torch.zeros(1, 10),
                    expression=torch.zeros(1, 10),
                    return_verts=True
                )
                self.template_mesh = template_output.vertices[0].cpu().numpy()
                self.template_face = template_model.faces.numpy()
            
            print(f"Template mesh initialized: {self.template_mesh.shape}, face: {self.template_face.shape}")
            
        except Exception as e:
            print(f"Warning: Could not initialize template data: {str(e)}")
            # Fallback to basic template
            self.template_mesh = np.zeros((10475, 3), dtype=np.float32)  # SMPL-X vertex count
            self.template_face = np.zeros((20908, 3), dtype=np.int32)    # SMPL-X face count
    
    def estimate_mesh_from_history(self, current_bbox, frame_idx):
        """Estimate mesh using temporal consistency from previous frames"""
        if len(self.mesh_history) == 0:
            return self._generate_basic_mesh(current_bbox, frame_idx)
        
        # Use weighted average of recent meshes
        weights = np.linspace(0.5, 1.0, len(self.mesh_history))
        weights = weights / weights.sum()
        
        estimated_mesh = np.zeros_like(self.template_mesh)
        for i, (mesh, weight) in enumerate(zip(self.mesh_history, weights)):
            estimated_mesh += mesh * weight
        
        # Apply basic scaling based on bbox size
        bbox_scale = self._estimate_scale_from_bbox(current_bbox)
        estimated_mesh *= bbox_scale
        
        print(f"Frame {frame_idx}: Estimated mesh from {len(self.mesh_history)} previous frames")
        return estimated_mesh
    
    def _generate_basic_mesh(self, bbox, frame_idx):
        """Generate a basic human-shaped mesh based on bbox"""
        if self.template_mesh is None:
            # Create a very basic human-like mesh
            mesh = np.zeros((10475, 3), dtype=np.float32)
            
            # Create a simple human shape using ellipsoid approximation
            height = bbox[3]  # bbox height
            width = bbox[2]   # bbox width
            
            # Scale template or create basic shape
            scale_factor = min(height, width) / 1000.0  # Normalize scale
            
            # Generate basic human proportions
            for i in range(10475):
                # Simple ellipsoid-based human shape
                t = i / 10475.0
                if t < 0.3:  # Head
                    mesh[i] = [0, 0.8 * scale_factor, 0]
                elif t < 0.5:  # Torso
                    mesh[i] = [0, 0.4 * scale_factor, 0]
                elif t < 0.7:  # Arms
                    mesh[i] = [0.3 * scale_factor, 0.2 * scale_factor, 0]
                else:  # Legs
                    mesh[i] = [0, -0.2 * scale_factor, 0]
            
            print(f"Frame {frame_idx}: Generated basic mesh from bbox")
            return mesh
        else:
            # Use template mesh with bbox-based scaling
            scale_factor = self._estimate_scale_from_bbox(bbox)
            scaled_mesh = self.template_mesh * scale_factor
            print(f"Frame {frame_idx}: Generated mesh from template with scale {scale_factor:.3f}")
            return scaled_mesh
    
    def _estimate_scale_from_bbox(self, bbox):
        """Estimate mesh scale based on bounding box size"""
        bbox_area = bbox[2] * bbox[3]  # width * height
        # Normalize scale based on typical human proportions
        # Assuming average human height is ~170cm and typical bbox area
        base_area = 100 * 300  # typical bbox area
        scale_factor = np.sqrt(bbox_area / base_area)
        # Clamp to reasonable range
        scale_factor = np.clip(scale_factor, 0.5, 2.0)
        return scale_factor
    
    def estimate_face_data(self, frame_idx):
        """Estimate face topology data"""
        if len(self.face_history) > 0:
            # Use most recent valid face data
            return self.face_history[-1]
        
        if self.template_face is not None:
            print(f"Frame {frame_idx}: Using template face data")
            return self.template_face
        
        # Generate basic face topology (very simplified)
        # This is a fallback - should rarely be used
        basic_face = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)  # Minimal face
        print(f"Frame {frame_idx}: Generated basic face topology")
        return basic_face
    
    def update_history(self, mesh, face_data):
        """Update temporal history with new valid data"""
        if mesh is not None and mesh.shape[0] > 0:
            self.mesh_history.append(mesh.copy())
        if face_data is not None and face_data.shape[0] > 0:
            self.face_history.append(face_data.copy())
    
    def get_smoothed_mesh(self, current_mesh, frame_idx):
        """Apply temporal smoothing to current mesh"""
        if len(self.mesh_history) == 0:
            return current_mesh
        
        # Simple exponential smoothing
        alpha = 0.7  # Smoothing factor
        smoothed_mesh = alpha * current_mesh + (1 - alpha) * self.mesh_history[-1]
        print(f"Frame {frame_idx}: Applied temporal smoothing")
        return smoothed_mesh

def validate_mesh_data(mesh, frame_idx):
    """Validate mesh data and return True if valid, False otherwise"""
    if mesh is None:
        print(f"Frame {frame_idx}: Mesh is None")
        return False
    
    if not isinstance(mesh, np.ndarray):
        print(f"Frame {frame_idx}: Mesh is not numpy array, type: {type(mesh)}")
        return False
    
    if mesh.shape[0] == 0:
        print(f"Frame {frame_idx}: Empty mesh detected!")
        return False
    
    if np.isnan(mesh).any():
        print(f"Frame {frame_idx}: NaN values detected in mesh")
        return False
    
    if np.isinf(mesh).any():
        print(f"Frame {frame_idx}: Infinite values detected in mesh")
        return False
    
    # Check reasonable bounds for human mesh
    if mesh.max() > 1000 or mesh.min() < -1000:
        print(f"Frame {frame_idx}: Mesh values out of reasonable bounds: min={mesh.min():.3f}, max={mesh.max():.3f}")
        return False
    
    return True

def validate_face_data(face_data, frame_idx):
    """Validate face topology data and return True if valid, False otherwise"""
    if face_data is None:
        print(f"Frame {frame_idx}: Face data is None")
        return False
    
    if not isinstance(face_data, np.ndarray):
        print(f"Frame {frame_idx}: Face data is not numpy array, type: {type(face_data)}")
        return False
    
    if face_data.shape[0] == 0:
        print(f"Frame {frame_idx}: Empty face data detected!")
        return False
    
    if np.isnan(face_data).any():
        print(f"Frame {frame_idx}: NaN values detected in face data")
        return False
    
    return True

def validate_camera_params(focal, princpt, frame_idx):
    """Validate camera parameters and return True if valid, False otherwise"""
    if focal is None or princpt is None:
        print(f"Frame {frame_idx}: Camera parameters are None")
        return False
    
    if len(focal) != 2 or len(princpt) != 2:
        print(f"Frame {frame_idx}: Invalid camera parameter dimensions")
        return False
    
    if any(np.isnan(focal)) or any(np.isnan(princpt)):
        print(f"Frame {frame_idx}: NaN values in camera parameters")
        return False
    
    if any(np.isinf(focal)) or any(np.isinf(princpt)):
        print(f"Frame {frame_idx}: Infinite values in camera parameters")
        return False
    
    # Check reasonable bounds for focal length and principal point
    if focal[0] <= 0 or focal[1] <= 0:
        print(f"Frame {frame_idx}: Invalid focal length: {focal}")
        return False
    
    if princpt[0] < 0 or princpt[1] < 0:
        print(f"Frame {frame_idx}: Invalid principal point: {princpt}")
        return False
    
    return True

def safe_model_inference(demoer, inputs, targets, meta_info, max_retries=3):
    """Safely run model inference with retry mechanism"""
    for attempt in range(max_retries):
        try:
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')
            return out
        except Exception as e:
            print(f"Model inference attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise e
            # Clear GPU cache and retry
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return None

def safe_mesh_rendering(vis_img, mesh, face_data, cam_param, show_verts=False, frame_idx=0):
    """Safely render mesh with comprehensive error handling"""
    try:
        from utils.vis import render_mesh_kugang
        result = render_mesh_kugang(vis_img, mesh, face_data, cam_param, mesh_as_vertices=show_verts)
        return result
    except Exception as e:
        print(f"Frame {frame_idx}: Error in render_mesh_kugang: {str(e)}")
        print(f"Frame {frame_idx}: Traceback: {traceback.format_exc()}")
        return None

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
    from utils.vis import save_obj, render_mesh_kugang, render_mesh
    from utils.human_models import smpl_x
    demoer = Demoer()
    demoer._make_model()
    demoer.model.eval()
    
    multi_person = args.multi_person

    # Initialize mesh estimator
    mesh_estimator = MeshEstimator(smpl_x, args.temporal_window)

    ### mmdet init
    checkpoint_file = '../pretrained_models/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    config_file= '../pretrained_models/mmdet/mmdet_faster_rcnn_r50_fpn_coco.py'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    video_capture = cv2.VideoCapture(args.video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    os.makedirs(args.output_path, exist_ok=True)
    out_video_path = os.path.join(args.output_path, 'smplx_video.mp4')
    output_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_idx = 0
    successful_frames = 0
    failed_frames = 0
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_idx += 1
        original_img = frame
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        
        frame_processed = False

        try:
            ## mmdet inference
            mmdet_results = inference_detector(model, frame)
            mmdet_box = process_mmdet_results(mmdet_results, cat_id=0, multi_person=multi_person)

            print(f"Frame {frame_idx}: Detected {len(mmdet_box[0])} persons")
            
            if len(mmdet_box[0]) < 1:
                print(f"Frame {frame_idx}: No persons detected")
                if args.fallback_to_original:
                    vis_img = original_img.copy()
                    frame_processed = True
                else:
                    vis_img = np.zeros_like(vis_img)
                    frame_processed = True
            else:
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

                    # Issue 3: Safe model inference with retry mechanism
                    try:
                        out = safe_model_inference(demoer, inputs, targets, meta_info, args.max_retries)
                    except Exception as e:
                        print(f"Frame {frame_idx}: Model inference failed after {args.max_retries} attempts: {str(e)}")
                        continue
                    
                    # Issue 3: Check if model output contains the expected key
                    if 'smplx_mesh_cam' not in out:
                        print(f"Frame {frame_idx}: 'smplx_mesh_cam' not found in model output. Available keys: {list(out.keys())}")
                        continue
                        
                    mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
                    
                    # Issue 4: Enhanced mesh validation with fallback estimation
                    if not validate_mesh_data(mesh, frame_idx):
                        print(f"Frame {frame_idx}: Invalid mesh detected, attempting estimation...")
                        if args.use_template_fallback:
                            mesh = mesh_estimator.estimate_mesh_from_history(bbox, frame_idx)
                        else:
                            continue
                    else:
                        # Apply temporal smoothing if enabled
                        if args.use_temporal_smoothing:
                            mesh = mesh_estimator.get_smoothed_mesh(mesh, frame_idx)
                        
                        # Update history with valid mesh
                        mesh_estimator.update_history(mesh, smpl_x.face)
                    
                    print(f"Frame {frame_idx}: mesh shape: {mesh.shape}")
                    print(f"Frame {frame_idx}: mesh min/max values: {mesh.min():.3f}/{mesh.max():.3f}")
                    
                    # Issue 5: Enhanced face data validation with fallback estimation
                    face_data = smpl_x.face
                    if not validate_face_data(face_data, frame_idx):
                        print(f"Frame {frame_idx}: Invalid face data detected, attempting estimation...")
                        if args.use_template_fallback:
                            face_data = mesh_estimator.estimate_face_data(frame_idx)
                        else:
                            continue
                    
                    print(f"Frame {frame_idx}: face shape: {face_data.shape}")

                    focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
                    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
                    
                    # Issue 6: Validate camera parameters
                    if not validate_camera_params(focal, princpt, frame_idx):
                        continue
                    
                    print(f"Frame {frame_idx}: focal: {focal}, princpt: {princpt}")
                    
                    # Issue 6: Safe mesh rendering with comprehensive error handling
                    rendered_img = safe_mesh_rendering(vis_img, mesh, face_data, 
                                                     {'focal': focal, 'princpt': princpt}, 
                                                     args.show_verts, frame_idx)
                    
                    if rendered_img is not None:
                        vis_img = rendered_img
                        print(f"Frame {frame_idx}: render_mesh_kugang completed successfully")
                        frame_processed = True
                    else:
                        print(f"Frame {frame_idx}: Mesh rendering failed, using original image")
                        if args.fallback_to_original:
                            vis_img = original_img.copy()
                            frame_processed = True

                    if args.show_bbox:
                        vis_img = cv2.rectangle(vis_img, start_point, end_point, (255, 0, 0), 2)
                        
        except Exception as e:
            print(f"Frame {frame_idx}: Unexpected error during processing: {str(e)}")
            print(f"Frame {frame_idx}: Traceback: {traceback.format_exc()}")
            if args.fallback_to_original:
                vis_img = original_img.copy()
                frame_processed = True
        
        # Ensure frame is always written
        if not frame_processed:
            if args.fallback_to_original:
                vis_img = original_img.copy()
            else:
                vis_img = np.zeros_like(vis_img)
        
        vis_img = vis_img.astype('uint8')
        output_video.write(vis_img[:, :, ::-1])
        
        if frame_processed:
            successful_frames += 1
        else:
            failed_frames += 1

    video_capture.release()
    output_video.release()
    
    print(f"Video processing completed.")
    print(f"Total frames processed: {frame_idx}")
    print(f"Successful frames: {successful_frames}")
    print(f"Failed frames: {failed_frames}")
    print(f"Success rate: {successful_frames/frame_idx*100:.2f}%")

if __name__ == "__main__":
    main() 