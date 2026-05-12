import torch
import torch.nn as nn
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output

# === ADD THIS IMPORT ===
from sam3.model.geometry_encoders import Prompt

class SAM3PromptTuningWrapper(nn.Module):
    def __init__(self, base_sam3_model, num_tokens=4):
        super().__init__()
        self.sam3 = base_sam3_model
        
        # 1. Freeze the ENTIRE base SAM3 model
        for param in self.sam3.parameters():
            param.requires_grad = False
            
        # 2. Define the learnable soft prompt
        embed_dim = 256 
        self.soft_prompt = nn.Parameter(torch.randn(num_tokens, 1, embed_dim))

    def forward(self, input_batch):
        """Intercepts the forward pass from trainer.py"""
        device = self.soft_prompt.device
        batch_size = len(input_batch.img_batch)
        seq_len = self.soft_prompt.shape[0]
        
        # Process image features normally
        backbone_out = {"img_batch_all_stages": input_batch.img_batch}
        backbone_out.update(self.sam3.backbone.forward_image(input_batch.img_batch))
        
        # INJECT SOFT PROMPT
        prompt_expanded = self.soft_prompt.expand(-1, batch_size, -1)
        backbone_out["language_features"] = prompt_expanded
        
        backbone_out["language_mask"] = torch.zeros(
            (batch_size, seq_len), dtype=torch.bool, device=device
        )
        backbone_out["language_embeds"] = prompt_expanded
        
        find_input = input_batch.find_inputs[0]
        find_target = input_batch.find_targets[0]
        # Extract the native, correctly batched geometric prompt from the dataset
        geometric_prompt = Prompt(
            box_embeddings=find_input.input_boxes,
            box_mask=find_input.input_boxes_mask,
            box_labels=find_input.input_boxes_label,
        )
        
        # Run standard SAM3 grounding, passing the geometric_prompt correctly
        out, hs = self.sam3.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=find_target,
            geometric_prompt=geometric_prompt
        )
        
        previous_stages_out = SAM3Output(iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE)
        previous_stages_out.append([out])
        
        return previous_stages_out, hs


def build_soft_prompt_sam3(**kwargs):
    """Hydra calls this. We let model_builder.py handle the heavy lifting."""
    num_tokens = kwargs.pop("num_tokens", 4)
    
    kwargs["checkpoint_path"] = "/home/kolomcon/data/sam3.pt"
    kwargs["load_from_HF"] = False
    kwargs["freeze_backbone"] = False 
    kwargs["eval_mode"] = False
    base_model = build_sam3_image_model(**kwargs)
    wrapped_model = SAM3PromptTuningWrapper(base_model, num_tokens=num_tokens)
    
    return wrapped_model