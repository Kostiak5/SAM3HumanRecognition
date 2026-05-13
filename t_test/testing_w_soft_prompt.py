import torch
import cv2
import numpy as np
import json
import os
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
import pycocotools.mask as mask_util
from tqdm import tqdm
# Import your custom builder
from sam3.custom_builder import build_soft_prompt_sam3
from sam3.train.transforms.basic_for_api import ComposeAPI
from sam3.train.transforms.segmentation import ResizeLongestSide, NormalizeImage, ToTensor
from t_test.testing_utils import determine_folders, parse_args, generate_colors, eval_set, visualize, select_keypoints, load_pts, load_ids

def load_trained_model(checkpoint_path):
    print("1. Building SAM3 + Soft Prompt Wrapper...")
    # Make sure eval_mode=True to disable the matcher and dropout layers!
    model = build_soft_prompt_sam3(num_tokens=4, eval_mode=True)
    model.eval()
    model.cuda()

    print(f"2. Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    
    # 3. Clean the DDP "module." prefixes if they exist
    state_dict = checkpoint.get("model", checkpoint) # Fallback just in case
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # 4. Inject the learned embeddings! 
    # strict=False allows it to load the soft_prompt even if the base SAM3 
    # weights aren't perfectly mapped in the checkpoint dictionary.
    model.load_state_dict(clean_state_dict, strict=False)
    
    print("3. Learned tokens successfully injected!")
    return model

def process_img(model, image_path):
    # 1. Load the image
    image_bgr = cv2.imread(image_path)
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Format the image exactly how SAM3 expects it
    # (Adjust these transforms to match your YAML's val_transforms)
    transform = ComposeAPI([
        ResizeLongestSide(target_length=1024),
        ToTensor(),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    input_dict = {"image": image_rgb}
    input_dict = transform(input_dict)
    
    # Move to GPU and add batch dimension (Batch Size = 1)
    img_tensor = input_dict["image"].unsqueeze(0).cuda()
    
    # 3. Create the dummy BatchedDatapoint (Mocking the dataloader)
    class DummyFindInput:
        def __init__(self):
            # Empty geometry! (This forces the model to use your soft prompt)
            self.input_boxes = torch.zeros((1, 0, 4), device="cuda")
            self.input_boxes_mask = torch.zeros((1, 0), dtype=torch.bool, device="cuda")
            self.input_boxes_label = torch.zeros((1, 0), dtype=torch.long, device="cuda")
            
    class DummyBatch:
        def __init__(self, img):
            self.img_batch = img
            self.find_inputs = [DummyFindInput()]
            self.find_targets = [None] # No ground truth during inference!

    batch = DummyBatch(img_tensor)
    
    print(f"4. Running inference on {image_path}...")
    with torch.no_grad():
        out, _ = model(batch)
        
    # 5. Extract the predictions
    final_stage_out = out[-1][0] 
    raw_masks = final_stage_out["pred_masks"] # Shape: (1, N, H, W)
    raw_scores = final_stage_out["pred_logits"] # Shape: (1, N, 1)
    
    # 6. Resize masks back to original image dimensions
    masks_resized = torch.nn.functional.interpolate(
        raw_masks,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False
    )
    
    # 7. Apply Sigmoid, Threshold, and DROP the Batch Dimension [0]
    masks = (masks_resized.sigmoid() > 0.5).cpu().numpy()[0] # Shape is now (N, orig_H, orig_W)
    scores = raw_scores.sigmoid().cpu().numpy()[0]           # Shape is now (N, 1)
    
    return scores, masks

def process_set(set_folder, set_out_folder=None, gt_folder=None, filename_to_id=None, id_to_kpts=None, args=None):
    eval_arr = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on: {device.type.upper()}")
    
    model = load_trained_model(CKPT_PATH)


    i = 0
    for img_path in tqdm(os.listdir(set_folder)):
        i += 1
        if i == 50:
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
        
        full_img_path = os.path.join(set_folder, img_path)
        scores, masks = process_img(model, full_img_path)
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

if __name__ == "__main__":
    # Point this to your best checkpoint (e.g., from Epoch 18 or 20)
    CKPT_PATH = "/home/kolomcon/data/SAM3_train_logs/checkpoints/checkpoint.pt"
    TEST_IMAGE = "/path/to/a/test/image.jpg"
     
    args = parse_args()
    SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER, KPTS_FOLDER = determine_folders(args)
    filename_to_id, id_to_kpts = load_ids(KPTS_FOLDER)
    id_to_kpts = load_pts(KPTS_FOLDER, id_to_kpts)

    process_set(SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER, filename_to_id, id_to_kpts, args)    
