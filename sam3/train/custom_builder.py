import torch
import torch.nn as nn
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output

class SAM3PromptTuningWrapper(nn.Module):
    def __init__(self, base_sam3_model, num_tokens=4):
        super().__init__()
        self.sam3 = base_sam3_model
        
        # 1. Freeze the ENTIRE base SAM3 model
        for param in self.sam3.parameters():
            param.requires_grad = False
            
        # 2. Define the learnable soft prompt
        # We know SAM3 uses a 256-dimensional space for cross-attention
        embed_dim = 256 
        self.soft_prompt = nn.Parameter(torch.randn(1, num_tokens, embed_dim))

    def forward(self, input_batch):
        """Intercepts the forward pass from trainer.py"""
        device = self.soft_prompt.device
        
        # Process image features normally
        backbone_out = {"img_batch_all_stages": input_batch.img_batch}
        backbone_out.update(self.sam3.backbone.forward_image(input_batch.img_batch))
        
        # INJECT SOFT PROMPT
        # We bypass the tokenizer and send our vector straight to the text transformer
        text_features = self.sam3.backbone.text_encoder.transformer(self.soft_prompt)
        backbone_out["language_features"] = text_features
        
        # Create a boolean mask of False (all learned tokens are valid)
        seq_len = self.soft_prompt.shape[1]
        backbone_out["language_mask"] = torch.zeros(
            (1, seq_len), dtype=torch.bool, device=device
        )
        
        find_input = input_batch.find_inputs[0]
        geometric_prompt = self.sam3._get_dummy_prompt(num_prompts=0)
        
        # Run standard SAM3 grounding
        out, hs = self.sam3.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt
        )
        
        previous_stages_out = SAM3Output(iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE)
        previous_stages_out.append([out])
        
        return previous_stages_out, hs


def build_soft_prompt_sam3(**kwargs):
    """Hydra calls this. We let model_builder.py handle the heavy lifting."""
    num_tokens = kwargs.pop("num_tokens", 4)
    
    # Force the native builder to load our local weights, not HuggingFace
    kwargs["checkpoint_path"] = "/home/kolomcon/data/sam3.pt"
    kwargs["load_from_HF"] = False
    kwargs["freeze_backbone"] = False # We handle the freezing manually!
    
    # 1. Build the base model (model_builder.py natively loads the weights here!)
    base_model = build_sam3_image_model(**kwargs)
    
    # 2. Wrap it (freezes all weights and adds the soft prompt)
    wrapped_model = SAM3PromptTuningWrapper(base_model, num_tokens=num_tokens)
    
    return wrapped_model

class SAM3ConvexPromptWrapper(nn.Module):
    def __init__(self, base_sam3_model):
        super().__init__()
        self.sam3 = base_sam3_model
        
        # Load the text anchors we extracted in Step 0!
        anchor_tensors = torch.load("anchor_tensors.pt")
        
        # Register them as a frozen buffer
        self.register_buffer("anchors", anchor_tensors)
        
        # Initialize trainable alpha weights
        num_anchors = anchor_tensors.shape[0]
        self.raw_alphas = nn.Parameter(torch.ones(num_anchors))
        self.soft_prompt = None
    
    def forward(self, input_batch):
        # 1. Enforce the Convex Constraint mathematically!
        # Softmax guarantees all alphas are positive and sum exactly to 1.0
        alphas = torch.softmax(self.raw_alphas, dim=0)
        
        # 2. Calculate the Convex Combination
        # Multiply each anchor by its weight and sum them up to create the single 256D prompt
        # Shape: (1, embed_dim)
        self.soft_prompt = torch.sum(alphas.view(-1, 1, 1) * self.anchors, dim=0, keepdim=True)
        device = self.soft_prompt.device
        
        # Process image features normally
        backbone_out = {"img_batch_all_stages": input_batch.img_batch}
        backbone_out.update(self.sam3.backbone.forward_image(input_batch.img_batch))
        
        # INJECT SOFT PROMPT
        # We bypass the tokenizer and send our vector straight to the text transformer
        text_features = self.sam3.backbone.text_encoder.transformer(self.soft_prompt)
        backbone_out["language_features"] = text_features
        
        # Create a boolean mask of False (all learned tokens are valid)
        seq_len = self.soft_prompt.shape[1]
        backbone_out["language_mask"] = torch.zeros(
            (1, seq_len), dtype=torch.bool, device=device
        )
        
        find_input = input_batch.find_inputs[0]
        geometric_prompt = self.sam3._get_dummy_prompt(num_prompts=0)
        
        # Run standard SAM3 grounding
        out, hs = self.sam3.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt
        )
        
        previous_stages_out = SAM3Output(iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE)
        previous_stages_out.append([out])
        
        return previous_stages_out, hs

def build_text_convcomb_prompt_sam3(**kwargs):
    """Hydra calls this. We let model_builder.py handle the heavy lifting."""
    
    # Force the native builder to load our local weights, not HuggingFace
    kwargs["checkpoint_path"] = "/home/kolomcon/data/sam3.pt"
    kwargs["load_from_HF"] = False
    kwargs["freeze_backbone"] = False # We handle the freezing manually!
    
    # 1. Build the base model (model_builder.py natively loads the weights here!)
    base_model = build_sam3_image_model(**kwargs)
    
    # 2. Wrap it (freezes all weights and adds the soft prompt)
    wrapped_model = SAM3ConvexPromptWrapper(base_model)
    
    return wrapped_model