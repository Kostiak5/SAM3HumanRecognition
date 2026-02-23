import torch
import os
import cv2
import numpy as np
import json
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import pycocotools.mask as mask_util
from tqdm import tqdm
from testing_utils import determine_folders, parse_args, generate_colors, eval_set, visualize, select_keypoints, load_ids, load_pts

import argparse
import os
COLORS = generate_colors(50)



def process_img(device, model, processor, img_folder, img_path, img_out_folder, pose_kpts_arr, text_prompt="human", args=None):
    logs = []

    # Load an image
    logs.append("Start processing")
    
    # Prompt the model with text
    logs.append("Image set")

    n_kpts = 6
    masks = []
    scores = []
    all_point_coords = []
    all_point_visibility = []
    for pose_kpts in pose_kpts_arr:
    # print(pose_kpts[:, :2][:n_kpts], pose_kpts[:, 2][:n_kpts])
        point_coords = pose_kpts[:, :2]
        point_visibility = pose_kpts[:, 2]
        point_coords_sorted, point_visibility_sorted, _ = select_keypoints(0.5, point_coords, point_visibility, method="distance+confidence")
        if point_visibility_sorted is None or len(point_visibility_sorted) < 6:
            continue
        all_point_coords.append(point_coords_sorted[:n_kpts])
        all_point_visibility.append(point_visibility_sorted[:n_kpts])
    return logs, np.array(all_point_coords), np.array(all_point_visibility)

def process_batch(masks_batch, scores_batch, all_point_coords, all_image_ids, all_image_paths, set_folder, set_out_folder, eval_arr, args):
    for masks, scores, point_coords, img_id, img_path in tqdm(zip(masks_batch, scores_batch, all_point_coords, all_image_ids, all_image_paths)):
        for mask, score in zip(masks, scores):
            mask_np = (mask[0] > 0.5).astype(np.uint8)
            # if mask_np.ndim == 3:
            #     mask_np = mask_np[0] # Take the first (and only) channel
            # 2. Encode with the correct 2D shape

            rle = mask_util.encode(np.asfortranarray(mask_np))
            if rle is not None:
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = rle['counts'].decode('utf-8')
                    eval_arr.append({
                        "segmentation": rle,
                        "score": float(score),
                        "image_id": int(img_id),
                        "category_id": 1
                    })
        if args.vis:
            image_out = visualize(
                    os.path.join(set_folder, img_path),
                    COLORS,
                    masks=masks,
                    scores=scores,
                    points=point_coords
                )
            cv2.imwrite(os.path.join(set_out_folder, img_path), image_out)
    

def process_set(set_folder, set_out_folder=None, gt_folder=None, filename_to_id=None, id_to_kpts=None, args=None):
    eval_arr = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on: {device.type.upper()}")
    
    model = build_sam3_image_model(enable_inst_interactivity=True)
    print(f"Model loaded on: {device.type.upper()}")
    model.to(device) 

    processor = Sam3Processor(model)
    print(f"Processor loaded")

    i = 0
    all_images = []
    all_image_ids = []
    all_point_coords = []
    all_point_visibility = []
    all_image_paths = []
    for img_path in tqdm(os.listdir(set_folder)):        
        i += 1
        if img_path[-3:] != "jpg":
            continue

        if filename_to_id is None:
            img_id = str(int(img_path[:-4]))
        else:
            img_id = filename_to_id[img_path]
        with Image.open(os.path.join(set_folder, img_path)) as image:
            all_images.append(image.convert("RGB"))
        all_image_paths.append(img_path)
        all_image_ids.append(img_id)
        _, point_coords, point_visibility = process_img(device, model, processor, set_folder, img_path, set_out_folder, id_to_kpts[img_id], args)
        print(point_coords, point_visibility)
        if point_coords is not None and point_coords.shape[0] > 0:
            all_point_coords.append(point_coords)
            all_point_visibility.append(np.ones_like(point_visibility))
        if i >= 1:
            if point_coords is not None and len(all_point_coords) > 0:
                inference_state = processor.set_image_batch(all_images)
                # inference_state = processor.set_text_prompt(state=inference_state, prompt="human") 
                masks_batch, scores_batch, _ = model.predict_inst_batch(
                    inference_state,
                    point_coords_batch=all_point_coords,
                    point_labels_batch=all_point_visibility,
                    multimask_output=False 
                )
                process_batch(masks_batch, scores_batch, all_point_coords, all_image_ids, all_image_paths, set_folder, set_out_folder, eval_arr, args)
            all_images = []
            all_image_ids = []
            all_point_coords = []
            all_point_visibility = []
            all_image_paths = []
            i = 0
    
    
    eval_set(eval_arr, gt_folder)


if __name__=="__main__":
    args = parse_args()
    SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER, KPTS_FOLDER = determine_folders(args)
    filename_to_id, id_to_kpts = load_ids(KPTS_FOLDER)
    id_to_kpts = load_pts(KPTS_FOLDER, id_to_kpts)

    process_set(SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER, filename_to_id, id_to_kpts, args)
