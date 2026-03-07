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
from t_test.testing_utils import determine_folders, parse_args, generate_colors, eval_set, visualize, select_keypoints, load_pts, load_ids

import argparse
import os
COLORS = generate_colors(50)

def process_img(device, model, processor, img_folder, img_path, img_out_folder, pose_kpts_arr, text_prompt="human", args=None):
    logs = []

    # Load an image
    logs.append("Start processing")
    image = Image.open(os.path.join(img_folder, img_path))
    inference_state = processor.set_image(image)
    # inference_state = processor.set_text_prompt(state=inference_state, prompt="potato")
    # Prompt the model with text
    logs.append("Image set")

    n_kpts = args.n_kpts
    masks = []
    scores = []
    all_point_coords = []
    for pose_kpts in pose_kpts_arr:
    # print(pose_kpts[:, :2][:n_kpts], pose_kpts[:, 2][:n_kpts])
        point_coords = pose_kpts[:, :2]
        point_visibility = pose_kpts[:, 2]
        point_coords_sorted, point_visibility_sorted, _ = select_keypoints(0.5, point_coords, point_visibility, method="distance+confidence")
        if point_visibility_sorted is None:
            continue
        # output_text = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        this_masks, this_scores, this_logits = model.predict_inst(
            inference_state,
            point_coords=point_coords_sorted[:n_kpts],
            point_labels=np.ones_like(point_visibility_sorted[:n_kpts]),
            multimask_output=False        )
        masks.append(this_masks)
        scores.append(this_scores)
        all_point_coords.append(point_coords_sorted[:n_kpts])

    
    # Get the masks, bounding boxes, and scores
    # masks, scores = output["masks"], output["boxes"], output["scores"]
    logs.append([scores])
    output = [masks, scores]
    if args is not None and args.vis:
        image_out = visualize(
                os.path.join(img_folder, img_path),
                COLORS,
                masks=masks,
                scores=scores,
                points=all_point_coords
            )
        cv2.imwrite(os.path.join(img_out_folder, img_path), image_out)
        logs.append(["Saved visualization: ", os.path.join(img_out_folder, img_path)])
    return logs, output

def process_set(set_folder, set_out_folder=None, gt_folder=None, filename_to_id=None, id_to_kpts=None, args=None):
    eval_arr = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on: {device.type.upper()}")
    
    model = build_sam3_image_model(enable_inst_interactivity=True)
    model.to(device) 

    processor = Sam3Processor(model)

    i = 0
    for img_path in tqdm(os.listdir(set_folder)):
        i += 1
        # if i == 100:
        #     break

        if img_path[-3:] != "jpg":
            continue

        if filename_to_id is None:
            img_id = str(int(img_path[:-4]))
        else:
            if img_path in filename_to_id:
                img_id = filename_to_id[img_path]
            else:
                continue

        _, output = process_img(device, model, processor, set_folder, img_path, set_out_folder, id_to_kpts[img_id], args=args)
        masks, scores = output
        # print(img_path, len(masks))
        if len(masks) == 0:
            continue
        

        for mask, score in zip(masks, scores):
            mask_np = mask.astype(np.uint8)
            if mask_np.ndim == 3:
                mask_np = mask_np[0] # Take the first (and only) channel
            
            # 2. Encode with the correct 2D shape
            rle = mask_util.encode(np.asfortranarray(mask_np))
            rle['counts'] = rle['counts'].decode('utf-8')
            eval_arr.append({
                "segmentation": rle,
                "score": float(score),
                "image_id": int(img_id),
                "category_id": 1
            })
    
    eval_set(eval_arr, gt_folder)

if __name__=="__main__":
    IMG_FOLDER = "t_test/test_images"
    IMG_PATH = "0000646.jpg"
    IMG_OUT_FOLDER = "t_test/test_images_out"
    args = parse_args()
    SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER, KPTS_FOLDER = determine_folders(args)
    filename_to_id, id_to_kpts = load_ids(KPTS_FOLDER)
    id_to_kpts = load_pts(KPTS_FOLDER, id_to_kpts)

    process_set(SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER, filename_to_id, id_to_kpts, args)
