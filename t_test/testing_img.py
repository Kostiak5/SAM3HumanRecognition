import torch
import os
import cv2
import numpy as np
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

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

def process_img(text_prompt="human"):
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    logs = []
    # Load an image
    IMG_FOLDER = "t_test/test_images"
    IMG_PATH = "0000646.jpg"
    IMG_OUT_FOLDER = "t_test/test_images_out"
    logs.append("Start processing")
    image = Image.open(os.path.join(IMG_FOLDER, IMG_PATH))
    inference_state = processor.set_image(image)
    # Prompt the model with text
    logs.append("Image set")

    output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    logs.append([boxes, scores])
    image_out = visualize(
            os.path.join(IMG_FOLDER, IMG_PATH),
            masks=masks,
            boxes=boxes,
            scores=scores
        )

    cv2.imwrite(os.path.join(IMG_OUT_FOLDER, IMG_PATH), image_out)
    logs.append(["Saved visualization: ", os.path.join(IMG_OUT_FOLDER, IMG_PATH)])
    return logs

if __name__=="__main__":
    logs = process_img()
    print(logs)
