import torch
import os
import cv2
import numpy as np
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import pycocotools.mask as mask_util

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


def visualize(image_path, boxes, scores, masks=None, score_threshold=0.3):
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
    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    if masks is not None:
        masks = masks.detach().cpu().numpy()

    n = boxes.shape[0]

    for i in range(n):
        score = scores[i]
        if score < score_threshold:
            continue

        # Pick unique color for this instance
        color = tuple(int(x) for x in COLORS[i % 50])  # RGB
        color_bgr = (color[2], color[1], color[0])     # Convert to BGR for OpenCV

        # Draw bounding box
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)

        # Draw score label
        cv2.putText(img, f"{score:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        # If mask exists → overlay with transparency
        if masks is not None:
            mask = masks[i]
            colored_mask = np.zeros_like(img)
            colored_mask[:, :, 0] = mask * color_bgr[0]
            colored_mask[:, :, 1] = mask * color_bgr[1]
            colored_mask[:, :, 2] = mask * color_bgr[2]

            # Alpha-blend mask
            img = cv2.addWeighted(img, 1.0, colored_mask, 0.4, 0)

    return img

def process_img(processor, img_folder, img_path, img_out_folder, text_prompt="human"):
    logs = []
    # Load an image
    logs.append("Start processing")
    image = Image.open(os.path.join(img_folder, img_path))
    inference_state = processor.set_image(image)
    # Prompt the model with text
    logs.append("Image set")

    output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    logs.append([boxes, scores])
    output = [masks, boxes, scores]

    image_out = visualize(
            os.path.join(img_folder, img_path),
            masks=masks,
            boxes=boxes,
            scores=scores
        )

    cv2.imwrite(os.path.join(img_out_folder, img_path), image_out)
    logs.append(["Saved visualization: ", os.path.join(img_out_folder, img_path)])
    return logs, output

def process_set(set_folder, set_out_folder=None, gt_folder=None):
    eval_arr = []

    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    for img_path in os.listdir(set_folder):
        _, output = process_img(processor, set_folder, img_path, set_out_folder)
        masks, boxes, scores = output
        if len(masks) == 0:
            continue
        
        
        img_id = str(int(img_path[:-4]))
        for mask, score in zip(masks, scores):
            rle = mask_util.encode(np.asfortranarray(mask.detach().cpu().numpy().astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('utf-8')
            eval_arr.append({
                "segmentation": rle,
                "score": float(score),
                "image_id": img_id,
                "category_id": 1
            })
    
    eval_set(eval_arr, gt_folder)

    

def determine_folders(subset):
    base = "../sam2.1/sam2"
    if subset == "COCO":
        set_folder = os.path.join(base,"COCO/original/val2017")
        gt_folder = os.path.join(base,"COCO/original/annotations/person_keypoints_val2017.json")
    elif subset == "OCHUMAN":
        set_folder = os.path.join(base,"OCHuman/COCO-like/val2017")
        gt_folder = os.path.join(base,"OCHuman/COCO-like/annotations/ochuman_coco_onlytest.json")
    elif subset == "CIHP":
        set_folder = os.path.join(base,"CIHP/val2017")
        gt_folder = os.path.join(base,"CIHP/annotations/person_keypoints_val2017.json")
    
    set_out_folder = os.path.join("../data", "SAM3_vis", "text_prompt", f"vis_{subset}")
    if not os.path.exists(set_out_folder):
        os.makedirs(set_out_folder)
    return set_folder, set_out_folder, gt_folder

if __name__=="__main__":
    IMG_FOLDER = "t_test/test_images"
    IMG_PATH = "0000646.jpg"
    IMG_OUT_FOLDER = "t_test/test_images_out"

    SUBSET = "COCO"

    SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER = determine_folders(SUBSET)

    process_set(SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER)
