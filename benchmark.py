import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import glob
from sklearn.metrics import roc_auc_score

# AltFreezing Imports (These will work perfectly here)
from config import config as cfg
from test_tools.common import detect_all, grab_all_frames
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader

# --- SETTINGS ---
max_frame = 10
cfg_path = "i3d_ori.yaml"
ckpt_path = "checkpoints/model.pth"

# Initialize Model (Once for the whole script)
cfg.init_with_yaml()
cfg.update_with_yaml(cfg_path)
cfg.freeze()

classifier = PluginLoader.get_classifier(cfg.classifier_type)()
classifier.cuda()
classifier.eval()
classifier.load(ckpt_path)

crop_align_func = FasterCropAlignXRay(cfg.imsize)
mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1, 1)
std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1, 1)

# --- EXACT DEMO.PY LOGIC WRAPPED IN A FUNCTION ---
def run_altfreezing_inference(video_path):
    # We bypass the cache saving/loading here to ensure fresh evaluation
    detect_res, all_lm68, frames = detect_all(video_path, return_frames=True, max_size=max_frame)
    
    if len(frames) == 0 or len(detect_res) == 0:
        return 0.5

    shape = frames[0].shape[:2]
    all_detect_res = []

    for faces, faces_lm68 in zip(detect_res, all_lm68):
        new_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            new_faces.append((box, lm5, face_lm68, score))
        all_detect_res.append(new_faces)

    detect_res = all_detect_res
    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)

    if len(tracks) == 0:
        tuples, tracks = find_longest(detect_res)
        
    if len(tracks) == 0:
        return 0.5

    data_storage = {}
    super_clips = []

    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        super_clips.append(len(track))
        for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
            box, lm5, lm68 = face[:3]
            big_box = get_crop_box(shape, box, scale=0.5)

            top_left = big_box[:2][None, :]
            new_lm5 = lm5 - top_left
            new_lm68 = lm68 - top_left
            new_box = (box.reshape(2, 2) - top_left).reshape(-1)
            info = (new_box, new_lm5, new_lm68, big_box)

            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]
            base_key = f"{track_i}_{j}_"
            data_storage[f"{base_key}img"] = cropped
            data_storage[f"{base_key}ldm"] = info

    clips_for_video = []
    clip_size = cfg.clip_size
    pad_length = clip_size - 1

    for super_clip_idx, super_clip_size in enumerate(super_clips):
        inner_index = list(range(super_clip_size))
        if super_clip_size < clip_size: 
            post_module = inner_index[1:-1][::-1] + inner_index
            l_post = len(post_module)
            post_module = post_module * (pad_length // l_post + 1)
            post_module = post_module[:pad_length]

            pre_module = inner_index + inner_index[1:-1][::-1]
            l_pre = len(pre_module)
            pre_module = pre_module * (pad_length // l_pre + 1)
            pre_module = pre_module[-pad_length:]

            inner_index = pre_module + inner_index + post_module

        super_clip_size = len(inner_index)
        frame_range = [inner_index[i : i + clip_size] for i in range(super_clip_size) if i + clip_size <= super_clip_size]
        for indices in frame_range:
            clips_for_video.append([(super_clip_idx, t) for t in indices])

    preds = []
    for clip in clips_for_video:
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        _, images_align = crop_align_func(landmarks, images)
        
        images_tensor = torch.as_tensor(images_align, dtype=torch.float32).cuda().permute(3, 0, 1, 2)
        images_tensor = images_tensor.unsqueeze(0).sub(mean).div(std)

        with torch.no_grad():
            output = classifier(images_tensor)
        pred = float(F.sigmoid(output["final_output"]))
        preds.append(pred)

    return np.mean(preds) if preds else 0.5# --- RUN EVALUATION ---
if __name__ == "__main__":
    # Updated to match your exact Kaggle folder structure
    FAKE_DIR = "/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23/FaceShifter/"
    REAL_DIR = "/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23/original/"

    print("Locating video files...")
    
    # Using recursive glob just in case there are minor subfolders
    real_videos = glob.glob(os.path.join(REAL_DIR, "**/*.mp4"), recursive=True)[:5]
    fake_videos = glob.glob(os.path.join(FAKE_DIR, "**/*.mp4"), recursive=True)[:5]
    
    print(f"Found {len(real_videos)} Real videos and {len(fake_videos)} Fake videos.")
    
    test_videos = real_videos + fake_videos
    labels = [0]*len(real_videos) + [1]*len(fake_videos)

    if len(real_videos) == 0 or len(fake_videos) == 0:
        print("🛑 ERROR: Could not find videos. Check the directory paths!")
    else:
        print(f"\nStarting native AltFreezing evaluation on {len(test_videos)} total videos...")
        
        predictions = []
        for vid in tqdm(test_videos, desc="Evaluating"):
            # print(f"[DEBUG] Processing: {os.path.basename(vid)}") # Uncomment if you want to see file names
            score = run_altfreezing_inference(vid)
            predictions.append(score)
            
        auc = roc_auc_score(labels, predictions)
        print(f"\n✅ Clean Baseline AUC: {auc:.4f}")
