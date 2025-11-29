import torch
import os
import cv2
import numpy as np
import argparse
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def parse_args():
    parser = argparse.ArgumentParser(description="Description of your script")
    # Dataset type. One of ("COCO", "OCHuman", "CrowdPose", "MPII", "AIC")
    parser.add_argument("--dataset", type=str, default="COCO")

    # Dataset subset. One of ("train", "val", "test")
    parser.add_argument("--subset", type=str, default="val")

    parser.add_argument("--checkpoint", type=str, default="sam2.1_hiera_base_plus.pt")

    # Number of images to process
    parser.add_argument("--num-images", type=int, default=10)
    args = parser.parse_args()

    # If boolean argument is not selected, set it to True
    for arg in args.__dict__:
        if args.__dict__[arg] is None:
            args.__dict__[arg] = True 

    # Check the dataset and subset
    assert args.dataset in ["COCO", "OCHuman", "OCHuman-tiny", "CrowdPose", "MPII", "AIC", "MMAPose", "SKV", "CIHP"]
    assert args.subset in ["train", "val", "test"]

    args.save_data_dir = None
    args.dataset_path = None
    args.gt_file = None
    if args.dataset == 'COCO':
        args.save_data_dir = 'COCO/original/sam_masks'
        args.dataset_path = f'../sam2/COCO/original/{args.subset}2017'
        args.gt_file = f'../sam2/COCO/original/annotations/person_keypoints_{args.subset}2017.json'
    elif args.dataset == 'CIHP':
        args.save_data_dir = 'CIHP/sam_masks'
        args.dataset_path = f'../sam2/CIHP/{args.subset}2017'
        args.gt_file = f'../sam2/CIHP/annotations/person_keypoints_{args.subset}2017.json'
    elif args.dataset == 'OCHuman':
        args.save_data_dir = 'OCHuman/COCO-like/sam_masks'
        args.dataset_path = f'../sam2/OCHuman/COCO-like/{args.subset}2017/'
        args.gt_file = f'../sam2/OCHuman/COCO-like/annotations/ochuman_coco_onlytest.json'
    return args


def generate_colors(n=50, seed=42):
    """Generate n distinct RGB colors."""
    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 255, size=(n, 3))
    return colors

COLORS = generate_colors(50)  # Pre-generate 50 distinct colors


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

def predict_set(args):
    for img in os.listdir(args.dataset_path):
        image = Image.open(os.path.join(IMG_FOLDER, IMG_PATH))
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt="human")

        # Prompt the model with text
        # Get the masks, bounding boxes, and scores
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        for i in range(masks.shape[0]):

if __name__=="__main__":
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    # Load an image
    IMG_FOLDER = "test"
    IMG_PATH = "0000646.jpg"
    IMG_OUT_FOLDER = "test/test_images_out"
    image = Image.open(os.path.join(IMG_FOLDER, IMG_PATH))
    inference_state = processor.set_image(image)
    # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt="human")

    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    print(boxes, scores)
    image_out = visualize(
            os.path.join(IMG_FOLDER, IMG_PATH),
            masks=masks,
            boxes=boxes,
            scores=scores
        )

    cv2.imwrite(os.path.join(IMG_OUT_FOLDER, IMG_PATH), image_out)
    print("Saved visualization: ", os.path.join(IMG_OUT_FOLDER, IMG_PATH))
