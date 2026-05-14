import torch
from sam3.custom_builder import build_soft_prompt_sam3

def extract_anchors():
    print("Loading base SAM3 model...")
    # Load the base model just to use its text encoder
    model = build_soft_prompt_sam3(num_tokens=4, eval_mode=True).cuda()
    
    # Define your anchor words
    words = ["human", "body", "crowd", "human body", "person", "man", "woman"]
    print(f"Extracting 256D embeddings for: {words}")
    
    with torch.no_grad():
        # Pass the strings through SAM3's text pipeline
        text_output = model.sam3.backbone.forward_text(words, device="cuda")
        
        # Extract the final mathematical representation
        # Shape will be roughly: (3, seq_len, 256)
        anchor_tensors = text_output["language_features"] 
        
    # Save the tensors to disk
    torch.save(anchor_tensors, "anchor_tensors.pt")
    print("Saved anchor_tensors.pt successfully!")

if __name__ == "__main__":
    extract_anchors()