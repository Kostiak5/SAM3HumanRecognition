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

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="SAM 3 Human Recognition and Visualization")

    parser.add_argument('--vis', action="store_true")

    return parser.parse_args()

def generate_colors(n=50, seed=42):
    """Generate n distinct RGB colors."""
    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 255, size=(n, 3))
    return colors

COLORS = generate_colors(50)  # Pre-generate 50 distinct colors

def eval_set(eval_arr, gt_folder):
    from xtcocotools.coco import COCO
    from xtcocotools.cocoeval import COCOeval

    cocoGt = COCO(gt_folder)
    cocoDt = cocoGt.loadRes(eval_arr)

    cocoEval = COCOeval(cocoGt, cocoDt, 'segm', sigmas=None, use_area=True)
    # if save_data_dir[:4] == "COCO":
    #     cocoEval.params.areaRng[0] = [1024, 1e10]

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print("evaluating, dumping ", cocoEval.stats[0].item())


def visualize(image_path, boxes=None, scores=None, masks=None, points=None, score_threshold=0.3):
    """
    image_path: path to a JPG image
    boxes: tensor Nx4 in XYXY format
    scores: tensor Nx1
    masks: optional tensor NxHxW (binary)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Move tensors to numpy on CPU
    if boxes is not None:
        boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    if masks is not None:
        masks = masks.detach().cpu().numpy()
    n = masks.shape[0]
    for i in range(n):
        score = scores[i]
        # print(masks[i], "mskshape")
        # Pick unique color for this instance
        color = tuple(int(x) for x in COLORS[i % 50])  # RGB
        color_bgr = (color[2], color[1], color[0])     # Convert to BGR for OpenCV

        # Draw bounding box
        if boxes is not None:
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)

            # Draw score label
            cv2.putText(img, f"{score:.2f}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        # If mask exists → overlay with transparency
        if masks is not None:
            mask = masks[i]
            if mask.ndim > 2:
                mask = np.squeeze(mask)
            print(mask.sum(), "vis")
            colored_mask = np.zeros_like(img)
            colored_mask[:, :, 0] = mask * color_bgr[0]
            colored_mask[:, :, 1] = mask * color_bgr[1]
            colored_mask[:, :, 2] = mask * color_bgr[2]
            mask_uint8 = np.ascontiguousarray((mask > 0.5).astype(np.uint8))
            # Alpha-blend mask
            img = cv2.addWeighted(img, 1.0, colored_mask, 0.4, 0)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw the outline fully opaque (thickness=2)
            cv2.drawContours(img, contours, -1, color_bgr, 2)

        if points is not None:
            # Reshape to (-1, 2) to handle both [N, 2] and [N, K, 2] point arrays
            for pt in points[i]:
                px, py = int(pt[0]), int(pt[1])
                # Skip if point is at (0,0) assuming it's a padded/invalid point
                if px == 0 and py == 0:
                    continue

                cv2.circle(img, (px, py), 6, (0, 0, 0), -1) 
                # 2. Middle white ring (radius 5)
                cv2.circle(img, (px, py), 5, (255, 255, 255), -1)
                # 3. Inner core using the instance's mask color (radius 3)
                cv2.circle(img, (px, py), 3, color_bgr, -1)

    return img

def select_keypoints(vis_thr, kpts, kpts_conf, ignore_limbs=False, method=None):
    "Implements different methods for selecting keypoints for pose2seg"

    methods = ["confidence", "distance", "distance+confidence", "distance*confidence", "closest"]
    assert method in methods, "Unknown method for selecting keypoints"

    limbs_id = [
            0, 0, 0,        # Face
            1, 1,           # Ears
            2, 2,           # Shoulders - body
            3, 4, 3, 4,     # Arms
            7, 7,           # Hips - body
            5, 6, 5, 6,     # Legs
        ]
    limbs_id = np.array(limbs_id)

    if method == "confidence":

        # Sort by confidence
        sort_idx = np.argsort(kpts_conf[kpts_conf >= vis_thr])[::-1]
        this_kpts = this_kpts[sort_idx, :2]
        kpts_conf = kpts_conf[sort_idx]

    elif method == "distance+confidence":

        if not ignore_limbs:
            facial_kpts = kpts[:3, :]
            facial_conf = kpts_conf[:3]
            facial_point = facial_kpts[np.argmax(facial_conf)]
            facial_point_conf = facial_conf[np.argmax(facial_conf)]
            if facial_point[-1] >= vis_thr:
                kpts = np.concatenate([facial_point[None, :], kpts[3:]], axis=0)
                kpts_conf = np.concatenate([[facial_point_conf], kpts_conf[3:]], axis=0)
                limbs_id = limbs_id[2:]

        # Ignore invisible keypoints 
        this_kpts = kpts[kpts_conf >= vis_thr, :2]
        kpts_conf = kpts_conf[kpts_conf >= vis_thr]

        # Sort by confidence

        sort_idx = np.argsort(kpts_conf[kpts_conf >= vis_thr])[::-1]
        this_kpts = this_kpts[sort_idx, :2]
        kpts_conf = kpts_conf[sort_idx]
        confidences = kpts_conf
        if confidences.shape[0] == 0:
            return this_kpts, None, np.random.permutation(this_kpts.shape[0])
        # Compute distance matrix between all pairs
        dist_matrix = np.linalg.norm(this_kpts[:, None, :2] - this_kpts[None, :, :2], axis=2)
        # First keypoint is the one with the highest confidence        
        selected_idx = [0]
        confidences[0] = -1
        for _ in range(this_kpts.shape[0] - 1):
            # Compute the distance to the closest selected keypoint
            min_dist = np.min(dist_matrix[:, selected_idx], axis=1)
            
            # Consider only keypoints with confidence in top 50%
            min_dist[confidences < np.percentile(confidences, 80)] = -1
            
            next_idx = np.argmax(min_dist)
            selected_idx.append(next_idx)
            confidences[next_idx] = -1

        this_kpts = this_kpts[selected_idx]
        kpts_conf = kpts_conf[selected_idx]

    return this_kpts, kpts_conf, selected_idx

def process_img(device, model, processor, img_folder, img_path, img_out_folder, pose_kpts_arr, text_prompt="human", args=None):
    logs = []

    # Load an image
    logs.append("Start processing")
    image = Image.open(os.path.join(img_folder, img_path))
    inference_state = processor.set_image(image)
    inference_state = processor.set_text_prompt(state=inference_state, prompt="human")
    # Prompt the model with text
    logs.append("Image set")

    n_kpts = 6
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

    # print(masks[0].sum())
    masks = torch.tensor(masks, device=device, dtype=torch.float32)
    scores = torch.tensor(scores, device=device, dtype=torch.float32)

    
    # Get the masks, bounding boxes, and scores
    # masks, scores = output["masks"], output["boxes"], output["scores"]
    logs.append([scores])
    output = [masks, scores]
    if args is not None and args.vis:
        image_out = visualize(
                os.path.join(img_folder, img_path),
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
            img_id = filename_to_id[img_path]

        _, output = process_img(device, model, processor, set_folder, img_path, set_out_folder, id_to_kpts[img_id], args)
        masks, scores = output
        # print(img_path, len(masks))
        if len(masks) == 0:
            continue
        elif len(masks.shape) == 1:
            masks = masks[None, 0]
            scores = scores[None, 0]
        

        for mask, score in zip(masks, scores):
            mask_np = mask.detach().cpu().numpy().astype(np.uint8)
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

    

def determine_folders(subset):
    base = "../sam2.1/sam2"
    if subset == "COCO":
        set_folder = os.path.join(base,"COCO/original/val2017")
        gt_folder = os.path.join(base,"COCO/original/annotations/person_keypoints_val2017.json")
        kpts_folder = os.path.join(base,"COCO/original/annotations/person_keypoints_val2017.json")

    elif subset == "OCHUMAN":
        set_folder = os.path.join(base,"OCHuman/COCO-like/val2017")
        gt_folder = os.path.join(base,"OCHuman/COCO-like/annotations/ochuman_coco_onlytest.json")
        kpts_folder = os.path.join(base,"COCO/original/annotations/person_keypoints_val2017.json")

    elif subset == "CIHP":
        set_folder = os.path.join(base,"CIHP/val2017")
        gt_folder = os.path.join(base,"CIHP/annotations/person_keypoints_val2017.json")
        kpts_folder = os.path.join(base,"CIHP/annotations/PMPose-b_GTmasks_CIHP_val.json")

    
    set_out_folder = os.path.join("../data", "SAM3_vis", "pt+text_prompt", f"vis_{subset}")
    if not os.path.exists(set_out_folder):
        os.makedirs(set_out_folder)
    return set_folder, set_out_folder, gt_folder, kpts_folder

def load_ids(gt_folder):
    with open(gt_folder, 'r') as f:
        data = json.load(f)
        filename_to_id = {img['file_name']: img['id'] for img in data['images']}
        id_to_kpts = {img['id']: [] for img in data['images']}
    
    return filename_to_id, id_to_kpts

def load_pts(gt_folder, id_to_kpts):
    with open(gt_folder, 'r') as f:
        data = json.load(f)
        for anno in data['annotations']:
            kpts = np.array(anno['keypoints']).reshape(-1, 3)
            vis = np.array(anno['visibility'])
            kpts[:, 2] = vis
            id_to_kpts[anno['image_id']].append(kpts)
    
    return id_to_kpts

if __name__=="__main__":
    IMG_FOLDER = "t_test/test_images"
    IMG_PATH = "0000646.jpg"
    IMG_OUT_FOLDER = "t_test/test_images_out"

    SUBSET = "CIHP"
    args = parse_args()
    SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER, KPTS_FOLDER = determine_folders(SUBSET)
    filename_to_id, id_to_kpts = load_ids(KPTS_FOLDER)
    id_to_kpts = load_pts(KPTS_FOLDER, id_to_kpts)

    process_set(SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER, filename_to_id, id_to_kpts, args)
