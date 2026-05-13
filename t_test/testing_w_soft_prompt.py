import torch
import os
import cv2
import numpy as np
import json
from PIL import Image
import pycocotools.mask as mask_util
from tqdm import tqdm

from sam3.custom_builder import build_soft_prompt_sam3
from sam3.model.sam3_image_processor import Sam3Processor
from t_test.testing_utils import determine_folders, parse_args, generate_colors, eval_set, visualize, select_keypoints, load_pts, load_ids

COLORS = generate_colors(50)

def load_trained_model(checkpoint_path):
    print("1. Building SAM3 + Soft Prompt Wrapper...")
    model = build_soft_prompt_sam3(num_tokens=4, eval_mode=True)
    model.eval()
    model.cuda()

    print(f"2. Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    state_dict = checkpoint.get("model", checkpoint) 
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)
    
    def override_forward_text(captions, input_boxes=None, additional_text=None, device="cuda"):
        # Match the batch size expected by the processor (usually 1 during inference)
        batch_size = len(captions)
        
        # In custom_builder.py, your soft prompt shape was initialized as (num_tokens, 1, embed_dim)
        # We just expand it to match the batch size: -> (num_tokens, batch_size, embed_dim)
        prompt_expanded = model.soft_prompt.expand(-1, batch_size, -1)
        
        # The sequence length is the number of tokens (4)
        seq_len = prompt_expanded.shape[0] 
        
        # Create a valid mask of False (meaning all tokens are valid and not padded)
        language_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        
        # Return exactly what SAM3 expects, completely bypassing the language model!
        return {
            "language_features": prompt_expanded,
            "language_mask": language_mask,
            "language_embeds": prompt_expanded
        }
        
    model.sam3.backbone.forward_text = override_forward_text
    print("3. Learned tokens successfully injected into Sam3Processor pipeline!")
    
    return model

def process_img(device, wrapped_model, processor, img_folder, img_path, img_out_folder, pose_kpts_arr, args=None):
    logs = []

    logs.append("Start processing")
    image = Image.open(os.path.join(img_folder, img_path))
    inference_state = processor.set_image(image)
    
    # Because we monkey-patched forward_text, it doesn't matter what string we pass here!
    # It will automatically bypass the text and use your learned soft prompt.
    inference_state = processor.set_text_prompt(state=inference_state, prompt="trigger_soft_prompt")
    logs.append("Image and Soft Prompt set")

    masks = inference_state["masks"].detach().cpu().numpy()
    scores = inference_state["scores"].detach().cpu().to(torch.float32).numpy()
    
    logs.append([scores])
    output = [masks, scores]
    
    if args is not None and args.vis:
        for i in range(masks.shape[0]):
            image_out = visualize(
                os.path.join(img_folder, img_path),
                COLORS,
                masks=masks,
                scores=scores
            )
            cv2.imwrite(os.path.join(img_out_folder, img_path), image_out)
        logs.append(["Saved visualization: ", os.path.join(img_out_folder, img_path)])
    return logs, output

def process_set(set_folder, set_out_folder=None, gt_folder=None, filename_to_id=None, id_to_kpts=None, args=None, ckpt_path=None):
    eval_arr = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on: {device.type.upper()}")
    
    # 1. Load the wrapped model
    wrapped_model = load_trained_model(ckpt_path)
    
    # 2. Pass the BASE model to Sam3Processor!
    # By passing wrapped_model.sam3, the processor gets the expected API, 
    # but still benefits from our monkey-patched forward_text logic.
    processor = Sam3Processor(wrapped_model.sam3)

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

        _, output = process_img(device, wrapped_model, processor, set_folder, img_path, set_out_folder, id_to_kpts[img_id], args=args)
        masks, scores = output
        
        if len(masks) == 0:
            continue

        for mask, score in zip(masks, scores):
            mask_np = mask.astype(np.uint8)
            if mask_np.ndim == 3:
                mask_np = mask_np[0] # Take the first (and only) channel
            
            # Encode with the correct 2D shape
            rle = mask_util.encode(np.asfortranarray(mask_np))
            rle['counts'] = rle['counts'].decode('utf-8')
            eval_arr.append({
                "segmentation": rle,
                "score": float(score),
                "image_id": int(img_id),
                "category_id": 1
            })
    
    if set_out_folder is not None:
        output_json_path = os.path.join(set_out_folder, f"output.json")

        with open(output_json_path, 'w') as f:
            json.dump(eval_arr, f)

        print(f"Segmentation masks saved to: {output_json_path}")
        
    eval_set(eval_arr, gt_folder)

if __name__=="__main__":
    # Point this to your best checkpoint
    CKPT_PATH = "/home/kolomcon/data/SAM3_train_logs/checkpoints/checkpoint.pt" 
    
    args = parse_args()
    SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER, KPTS_FOLDER = determine_folders(args)
    filename_to_id, id_to_kpts = load_ids(KPTS_FOLDER)
    id_to_kpts = load_pts(KPTS_FOLDER, id_to_kpts)

    process_set(SET_FOLDER, SET_OUT_FOLDER, GT_FOLDER, filename_to_id, id_to_kpts, args, ckpt_path=CKPT_PATH)