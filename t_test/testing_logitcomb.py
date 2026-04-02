import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np
import json
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import pycocotools.mask as mask_util
from tqdm import tqdm
from t_test.testing_utils import determine_folders, parse_args, generate_colors, eval_set, visualize, select_keypoints, load_pts, load_ids, load_pts_bboxes, compress_logits
import copy
import argparse
import os
COLORS = generate_colors(50)

def assign_mask_to_kpts(masks, kpts):
    mask_i = -1
    for i, mask in enumerate(masks):
        n_kpts_inside_mask = 0
        for kpt in kpts:
            if 0 < round(kpt[1]) < mask.shape[0] and 0 < round(kpt[0]) < mask.shape[1] and mask[round(kpt[1])][round(kpt[0])] == 1:
                n_kpts_inside_mask += 1

                if n_kpts_inside_mask >= 4:
                    print("Found")
                    break
        if n_kpts_inside_mask >= 4:
            print("Found later")
            return i
        elif n_kpts_inside_mask > 0:
            mask_i = i
    print("Not found:", mask_i)
    return mask_i
    
    

def process_img(device, model, processor, img_folder, img_path, img_out_folder, pose_kpts_arr, bbox_arr, text_prompt="person", args=None):
    logs = []

    # Load an image
    logs.append("Start processing")
    image = Image.open(os.path.join(img_folder, img_path))
    imgw, imgh = image.size
    inference_state = processor.set_image(image)
    # inference_state = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    # Prompt the model with text
    logs.append("Image set")

    n_kpts = args.n_kpts
    masks = []
    scores = []
    
    base_gl_state = copy.deepcopy(inference_state)

    for idx, (pose_kpts, bbox) in enumerate(zip(pose_kpts_arr, bbox_arr)):
        all_point_coords = []    
        # print(pose_kpts[:, :2][:n_kpts], pose_kpts[:, 2][:n_kpts])
        base_state = copy.deepcopy(base_gl_state)
        point_coords = pose_kpts[:, :2]
        point_visibility = pose_kpts[:, 2]
        point_coords_sorted, point_visibility_sorted, _ = select_keypoints(0.5, point_coords, point_visibility, method="distance+confidence")
        if point_visibility_sorted is None or len(point_visibility_sorted) == 0:
            continue
        
        normalized_pt = copy.deepcopy(point_coords_sorted[:n_kpts])
        normalized_pt[:, 0] /= imgw
        normalized_pt[:, 1] /= imgh

        for i in range(1):
            # base_state = processor.add_point_prompt(state=base_state, point=normalized_pt[i], label=1)
            base_state = processor.set_text_prompt(state=base_state, prompt=text_prompt)
            # point_coords_cropped = point_coords_sorted[:(i+1)]
            # if len(point_coords_cropped.shape) < 2:
            #     point_coords_cropped = [point_coords_cropped]
            # image_out = visualize(
            #             os.path.join(img_folder, img_path),
            #             COLORS,
            #             masks=base_state["masks"].cpu().detach().numpy(),
            #             scores=base_state["scores"].cpu().detach().to(torch.float32).numpy(),
            #             points=point_coords_sorted[:(i+1)]
            #         )
            # cv2.imwrite(os.path.join(img_out_folder, f"{img_path}_{idx}_{i}.jpg"), image_out)
            this_masks, this_scores, pvs_logits = model.predict_inst(
                inference_state,
                point_coords=point_coords_sorted[:n_kpts],
                point_labels=np.ones_like(point_visibility_sorted[:n_kpts]),
                multimask_output=False
            )
            if pvs_logits is not None and base_state["masks_logits"] is not None:
                pcs_logits = base_state["masks_logits"].cpu().detach().numpy()
                resized_pvs_logits = F.interpolate(
                    pvs_logits.unsqueeze(0), 
                    size=(pcs_logits.shape[2], pcs_logits.shape[3]), 
                    mode='bilinear', 
                    align_corners=False
                )
                clamped_pvs_logits = np.clip(resized_pvs_logits, 0.0, 1.0)
                squared_diff = (pcs_logits - clamped_pvs_logits) ** 2
    
                # 2. Average the error across Channels, Height, and Width
                # axis=(-3, -2, -1) ensures we get one distance value per candidate
                mse_distances = np.mean(squared_diff, axis=(-3, -2, -1))
                
                # 3. Find the index of the minimum distance
                closest_idx = np.argmin(mse_distances)

                pcs_best_logits = pcs_logits[closest_idx]
                print(pcs_best_logits.shape)

                combined_logits = (clamped_pvs_logits[0] + pcs_best_logits) * 0.5
                best_mask = combined_logits > 0.5
                this_masks = best_mask                     
            
        if 'scores' in base_state and len(base_state['scores']) != 0:
            # this_masks = base_state["masks"].cpu().detach().numpy()
            # this_scores = base_state["scores"].cpu().detach().to(torch.float32).numpy()
            # max_score_mask = np.argmax(this_scores)
            masks.append(this_masks[0])
            scores.append(this_scores[0])
            if args is not None and args.vis and base_state["masks"] is not None and len(base_state["masks"]) > 0:
                print(base_state["masks"].cpu().detach().numpy().shape)
                image_out = visualize(
                        os.path.join(img_folder, img_path),
                        COLORS,
                        masks=np.array([this_masks[0]]),
                        scores=np.array([this_scores[0]]),
                        points=[point_coords_sorted[0]]
                    )
                cv2.imwrite(os.path.join(img_out_folder, f"{img_path}_{idx}.jpg"), image_out)
                logs.append(["Saved visualization: ", os.path.join(img_out_folder, img_path)])

    
    # Get the masks, bounding boxes, and scores
    # masks, scores = output["masks"], output["boxes"], output["scores"]
    logs.append([scores])
    output = [masks, scores]
    return logs, output


def process_set(set_folder, set_out_folder=None, gt_folder=None, filename_to_id=None, id_to_kpts=None, id_to_bboxes=None, args=None):
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

        _, output = process_img(device, model, processor, set_folder, img_path, set_out_folder, id_to_kpts[img_id], id_to_bboxes[img_id], args=args)
        masks, scores = output
        # print(img_path, len(masks))
        if len(masks) == 0:
            continue
        

        for mask, score in zip(masks, scores):
            print(mask.shape, score.shape)
            if mask is not None and mask.any():
                # Ensure mask is 2D and uint8
                mask_np = np.squeeze(mask).astype(np.uint8)
                rle_list = mask_util.encode(np.asfortranarray(mask_np))
                rle = rle_list[0] if isinstance(rle_list, list) else rle_list

                # 3. Decode the bytes to string
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = rle['counts'].decode('utf-8')
                eval_arr.append({
                    "segmentation": rle,
                    # "score": float(score),
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
    id_to_kpts, id_to_bboxes = load_pts_bboxes(KPTS_FOLDER, id_to_kpts)

    process_set(SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER, filename_to_id, id_to_kpts, id_to_bboxes, args)
