import os
import torch
import numpy as np

from PIL import Image
import cv2

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from utils.intrinsic_warmup import warmup_intrinsics_linear, warmup_intrinsics_fix

def VGGT_estimation(model, target_size, video_pth, extrinsic_saved_pth, intrinsic_saved_pth, device, warmup_intrinsics=False, return_estimated_depth=False):
    cap = cv2.VideoCapture(video_pth)
    pil_frames = []
    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR → RGB
        img = Image.fromarray(frame) # 转换为 PIL.Image
        if index % 1 == 0:
            pil_frames.append(img)
        index += 1
    cap.release()
    print(f"len(pil_frames): {len(pil_frames)}")
    
    images = load_and_preprocess_images(pil_frames, target_size=target_size).to(device)
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
            predictions = model(images)
    
    pose_enc = predictions['pose_enc']
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    if warmup_intrinsics:
        intrinsic = warmup_intrinsics_fix(intrinsic.squeeze(0), warmup=10).unsqueeze(0)
    
    output_depth = []
    if return_estimated_depth:
        depth = predictions['depth']
        output_depth = []
        for i in range(depth.shape[1]):
            depth_mean = depth[0, i, :, :, 0]
            depth_mean = depth_mean.cpu().detach().numpy()
            depth_mean = depth_mean.mean()
            output_depth.append(depth_mean)
    else:
        output_depth.append(0.5) # Fix depth value.
    
    try:
        torch.save(extrinsic, extrinsic_saved_pth)
        torch.save(intrinsic, intrinsic_saved_pth)
        success_saved_camera_poses = True
    except:
        success_saved_camera_poses = False

    return extrinsic, intrinsic, success_saved_camera_poses, output_depth

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run a local VGGT camera estimation smoke test.")
    parser.add_argument("--model-path", type=str, default="checkpoints/VGGT-1B")
    parser.add_argument("--image-dir", type=str, default="assets/examples/frames")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained(args.model_path).to(device)
    
    image_dir = args.image_dir
    
    video_name = image_dir.split("/")[-1]
    image_names = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                image_names.append(os.path.join(root, file))
    image_names.sort()
    
    images = load_and_preprocess_images(image_names, target_size=518).to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)

    # keys(): dict_keys(['pose_enc', 'pose_enc_list', 'depth', 'depth_conf', 'world_points', 'world_points_conf']) 
    pose_enc = predictions['pose_enc']
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    torch.save(extrinsic, f"{args.saved_trinsic_pth}/extrinsic_{video_name}.pt")
    torch.save(intrinsic, f"{args.saved_trinsic_pth}/intrinsic_{video_name}.pt")