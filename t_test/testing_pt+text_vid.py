import torch
import os
import cv2
import numpy as np
import json
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_video_predictor
import pycocotools.mask as mask_util
from tqdm import tqdm
from t_test.testing_utils import determine_folders, parse_args, generate_colors, eval_set, visualize, select_keypoints, load_pts, load_ids, load_pts_bboxes, compress_logits
import copy
import argparse
import os
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)
from t_test.compare_to_gt import GT
COLORS = generate_colors(50)

def assign_mask_to_kpts(masks, kpts):
    mask_i = -1
    n_kpts_inside_mask = 0
    for mask_i, mask in enumerate(masks):
        print(f"Assigning: {mask.sum()} vs {kpts}")
        for kpt in kpts:
            if 0 < round(kpt[1]) < mask.shape[0] and 0 < round(kpt[0]) < mask.shape[1] and mask[round(kpt[1])][round(kpt[0])] == 1:
                n_kpts_inside_mask += 1

                if n_kpts_inside_mask >= 3:
                    print("Found", mask_i)
                    return mask_i

    print("Not found")

    return -1
    
def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")

def process_img(device, predictor, img_folder, img_path, img_out_folder, instance_arr, text_prompt="person", args=None):
    logs = []
    # Load an image
    logs.append("Start processing")
    image = Image.open(os.path.join(img_folder, img_path))
    imgw, imgh = image.size
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=os.path.join(img_folder, img_path),
        )
    )
    session_id = response["session_id"]
    frame_idx = 0
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=text_prompt,
        )
    )
    out = response["outputs"]

    visualize_formatted_frame_output(
        frame_idx,
        [os.path.join(img_folder, img_path)],
        outputs_list=[prepare_masks_for_visualization({frame_idx: out})],
        titles=["SAM 3 Dense Tracking outputs"],
        figsize=(6, 4),
        output_path=os.path.join(img_out_folder, f"{img_path}_text.jpg")
    )

    # print(f"out keys: {out}")
    n_kpts = args.n_kpts
    masks = []
    scores = []
    
    for idx, instance in enumerate(instance_arr):
        pose_kpts = instance['keypoints']
        bbox = instance['bbox']
        if 'id' in instance:
            inst_id = instance['id']
        all_point_coords = []    
        # print(pose_kpts[:, :2][:n_kpts], pose_kpts[:, 2][:n_kpts])
        # base_state = copy.deepcopy(base_gl_state)
        point_coords = pose_kpts[:, :2]
        point_visibility = pose_kpts[:, 2]
        point_coords_sorted, point_visibility_sorted, _ = select_keypoints(0.5, point_coords, point_visibility, method="distance+confidence")
        if point_visibility_sorted is None or len(point_visibility_sorted) == 0:
            continue


        points_tensor = torch.tensor(
            abs_to_rel_coords(point_coords_sorted[:n_kpts], imgw, imgh, coord_type="point"),
            dtype=torch.float32,
        )
        points_labels_tensor = torch.ones(points_tensor.shape[0], dtype=torch.int32)
        obj_id = assign_mask_to_kpts(out['out_binary_masks'], point_coords_sorted[:n_kpts])
        if obj_id >= 0:
            response = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_idx,
                    points=points_tensor,
                    point_labels=points_labels_tensor,
                    obj_id=obj_id,
                )
            )

            if inst_id >= 0:
                new_out = response["outputs"]
                # print(f"IoU w text only: {GT_EVALUATOR.iou_to_gt(inst_id, out['out_binary_masks'][obj_id])}")
                # print(f"IoU w text+pt: {GT_EVALUATOR.iou_to_gt(inst_id, new_out['out_binary_masks'][obj_id])}")

        if args.vis and args.vis_folder is not None:
            visualize_formatted_frame_output(
                frame_idx,
                [os.path.join(img_folder, img_path)],
                outputs_list=[prepare_masks_for_visualization({frame_idx: out})],
                titles=["SAM 3 Dense Tracking outputs"],
                figsize=(6, 4),
                output_path=os.path.join(img_out_folder, f"{img_path}_text.jpg")
            )
        # if 'scores' in base_state and len(base_state['scores']) != 0:
        #     # this_masks = base_state["masks"].cpu().detach().numpy()
        #     # this_scores = base_state["scores"].cpu().detach().to(torch.float32).numpy()
        #     # max_score_mask = np.argmax(this_scores)
        #     masks.append(this_masks[0])
        #     scores.append(this_scores[0])
        #     if args is not None and args.vis and base_state["masks"] is not None and len(base_state["masks"]) > 0:
        #         print(base_state["masks"].cpu().detach().numpy().shape)
        #         image_out = visualize(
        #                 os.path.join(img_folder, img_path),
        #                 COLORS,
        #                 masks=np.array([this_masks[0]]),
        #                 scores=np.array([this_scores[0]]),
        #                 points=[point_coords_sorted[0]]
        #             )
        #         cv2.imwrite(os.path.join(img_out_folder, f"{img_path}_{idx}.jpg"), image_out)
        #         logs.append(["Saved visualization: ", os.path.join(img_out_folder, img_path)])

    
    # Get the masks, bounding boxes, and scores
    # masks, scores = output["masks"], output["boxes"], output["scores"]
    final_masks = response["outputs"]['out_binary_masks']
    final_scores = response["outputs"]['out_probs']
    logs.append([scores])
    print(f"final scores: {final_scores}")
    output = [final_masks, final_scores]

    _ = predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    return logs, output


def process_set(set_folder, set_out_folder=None, gt_folder=None, filename_to_id=None, id_to_instance=None, args=None):
    eval_arr = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on: {device.type.upper()}")
    
    predictor = build_sam3_video_predictor(gpus_to_use=[torch.cuda.current_device()])

    i = 0
    for img_path in tqdm(os.listdir(set_folder)):
        i += 1
        if i == 100:
            break

        if img_path[-3:] != "jpg":
            continue

        if filename_to_id is None:
            img_id = str(int(img_path[:-4]))
        else:
            if img_path in filename_to_id:
                img_id = filename_to_id[img_path]
            else:
                continue

        _, output = process_img(device, predictor, set_folder, img_path, set_out_folder, id_to_instance[img_id], args=args)
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
    predictor.shutdown()
    GT_EVALUATOR.eval_set(eval_arr)

if __name__=="__main__":
    IMG_FOLDER = "t_test/test_images"
    IMG_PATH = "0000646.jpg"
    IMG_OUT_FOLDER = "t_test/test_images_out"
    args = parse_args()
    SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER, KPTS_FOLDER = determine_folders(args)
    filename_to_id, id_to_instance = load_ids(KPTS_FOLDER)
    id_to_instance = load_pts_bboxes(KPTS_FOLDER, id_to_instance)
    GT_EVALUATOR = GT(GT_FOLDER)
    process_set(SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER, filename_to_id, id_to_instance, args)
