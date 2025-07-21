from annotator.dwpose import DWposeDetector
import math
import argparse
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
import pickle
import subprocess
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, default='', help = 'input path')
parser.add_argument('--output_path', type=str, default='out_demo', help = 'output path')
args = parser.parse_args()

def extract_frames_with_ffmpeg(video_path, output_dir):
    """Extract frames using ffmpeg as a fallback when OpenCV fails"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get frame count using ffprobe
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', 
           '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    frame_count = int(result.stdout.strip())
    
    # Extract frames
    cmd = ['ffmpeg', '-i', video_path, '-vf', 'fps=30', '-frame_pts', '1', 
           os.path.join(output_dir, 'frame_%06d.jpg'), '-y']
    subprocess.run(cmd, capture_output=True)
    
    # Read extracted frames
    frames = []
    for i in range(1, frame_count + 1):
        frame_path = os.path.join(output_dir, f'frame_{i:06d}.jpg')
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)
    
    return frames

if __name__ == "__main__":
    video_path = args.video_path
    assert os.path.exists(video_path)
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    pose = DWposeDetector()
    
    # Try OpenCV first
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {frame_count} frames")
    
    # Test if we can read the first frame
    ret, test_frame = cap.read()
    if not ret:
        print("OpenCV cannot read frames, using ffmpeg fallback...")
        cap.release()
        
        # Use ffmpeg to extract frames
        temp_dir = os.path.join(output_path, 'temp_frames')
        frames = extract_frames_with_ffmpeg(video_path, temp_dir)
        print(f"Extracted {len(frames)} frames using ffmpeg")
        
        # Process frames
        results = []
        for frame in tqdm(frames, desc="Processing frames"):
            results.append(pose(frame))
            
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir)
        
    else:
        print("Using OpenCV for frame reading...")
        cap.release()
        cap = cv2.VideoCapture(video_path)
        results = []
        for _ in tqdm(range(frame_count), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret: 
                break
            results.append(pose(frame))
        cap.release()

    print(f"Saving {len(results)} pose detection results...")
    with open(os.path.join(output_path, 'dwpose_video.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("Done!") 