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
from t_test.testing_utils import determine_folders, parse_args, generate_colors, eval_set, visualize, select_keypoints, load_pts, load_ids, load_pts_bboxes
from t_test.compare_to_gt import mask_to_binary, GT

import argparse
import os
COLORS = generate_colors(50)

def process_img(device, model, processor, img_folder, img_path, img_out_folder, inst_arr, text_prompt="human", args=None, gt_evaluator=None):
    logs = []

    # Load an image
    logs.append("Start processing")
    image = Image.open(os.path.join(img_folder, img_path))
    imgw, imgh = image.size
    inference_state = processor.set_image(image)
    # inference_state = processor.set_text_prompt(state=inference_state, prompt="potato")
    # Prompt the model with text
    logs.append("Image set")

    n_kpts = args.n_kpts
    masks = []
    scores = []
    all_point_coords = []
    for inst in inst_arr:
    # print(pose_kpts[:, :2][:n_kpts], pose_kpts[:, 2][:n_kpts])
        pose_kpts = inst['keypoints']
        segm = inst['segmentation']
        point_coords = pose_kpts[:, :2]
        point_visibility = pose_kpts[:, 2]
        
        point_coords_sorted, point_visibility_sorted, _ = select_keypoints(
            0.5, point_coords, point_visibility, method="distance+confidence"
        )
        if point_visibility_sorted is None:
            continue
        
        this_masks, this_scores, this_logits = model.predict_inst(
            inference_state,
            point_coords=point_coords_sorted[:n_kpts],
            point_labels=np.ones_like(point_visibility_sorted[:n_kpts]),
            multimask_output=False        )
        ## CROP
        #         gt_bin_mask = mask_to_binary(segm, imgh, imgw)
        # rows = np.any(gt_bin_mask, axis=1)
        # cols = np.any(gt_bin_mask, axis=0)
        
        # # Safety check: skip if mask is entirely empty
        # if not np.any(rows) or not np.any(cols):
        #     continue
            
        # ymin, ymax = np.where(rows)[0][[0, -1]]
        # xmin, xmax = np.where(cols)[0][[0, -1]]
        
        # # Crop the PIL image (right and lower bounds are exclusive in PIL, so +1)
        # cropped_image = image.crop((xmin, ymin, xmax + 1, ymax + 1))
        
        # # Set the inference state using the newly cropped image
        # inference_state = processor.set_image(cropped_image)
        # # ---------------------------------------------------------

    
        
       
            
        # # ---------------------------------------------------------
        # # 2. SHIFT KEYPOINTS TO MATCH CROPPED COORDINATES
        # # ---------------------------------------------------------
        # # Subtract the bounding box starting point (xmin, ymin) from the coordinates
        # adjusted_coords = point_coords_sorted[:n_kpts] - np.array([[xmin, ymin]])
        # output_text = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        # this_masks, this_scores, this_logits = model.predict_inst(
        #     inference_state,
        #     point_coords=adjusted_coords,
        #     point_labels=np.ones_like(point_visibility_sorted[:n_kpts]),
        #     multimask_output=False        )
       
        # this_masks_np = np.array(this_masks) # Ensure it's a numpy array for slicing
    
        # # Create an empty array of the original image size, keeping batch/channel dims
        # # this_masks_np is usually shape (1, 1, crop_h, crop_w) or (1, crop_h, crop_w)
        # target_shape = list(this_masks_np.shape)
        # target_shape[-2] = imgh # Replace height
        # target_shape[-1] = imgw # Replace width
        
        # restored_mask = np.zeros(target_shape, dtype=this_masks_np.dtype)
        
        # # Paste the predicted mask into the correct location in the full-size array
        # restored_mask[..., ymin:ymax+1, xmin:xmax+1] = this_masks_np
        ## CROP end

        # if gt_evaluator is not None:
        #     print("IoU of mask :", gt_evaluator.iou_to_gt(inst['id'], restored_mask[0]))

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

def process_set(set_folder, set_out_folder=None, gt_folder=None, gt_evaluator=None, filename_to_id=None, id_to_instance=None, args=None):
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

        _, output = process_img(device, model, processor, set_folder, img_path, set_out_folder, id_to_instance[img_id], args=args, gt_evaluator=gt_evaluator)
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
    
    if SET_OUT_FOLDER is not None:
        output_json_path = os.path.join(SET_OUT_FOLDER, f"output.json")

        with open(output_json_path, 'w') as f:
            json.dump(eval_arr, f)

        print(f"Segmentation masks saved to: {output_json_path}")
        
    eval_set(eval_arr, gt_folder)

if __name__=="__main__":
    IMG_FOLDER = "t_test/test_images"
    IMG_PATH = "0000646.jpg"
    IMG_OUT_FOLDER = "t_test/test_images_out"
    args = parse_args()
    SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER, KPTS_FOLDER = determine_folders(args)
    filename_to_id, id_to_instance = load_ids(KPTS_FOLDER)
    id_to_instance = load_pts_bboxes(KPTS_FOLDER, id_to_instance)
    GT_EVALUATOR = GT(GT_FOLDER)
    process_set(SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER, GT_EVALUATOR, filename_to_id, id_to_instance, args)
