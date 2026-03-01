import torch
import cv2
import os
import sys
from pathlib import Path
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights, mobilenet_v3_small

from infrastructure.ml.models.vit_ctc.model import ModelTextRecoginizer
from infrastructure.ml.models.vit_ctc.pos_patch_embed import PatchEmbedWithPos
from infrastructure.ml.models.vit_ctc.patch_embedding import PatchEmbedding
from infrastructure.ml.models.vit_ctc.position_embedding import PosEmbedding
from infrastructure.ml.models.vit_ctc.backbone import Backbone
from infrastructure.ml.models.vit_ctc.transformer import TransformerBlock

def create_model():
    rows = 1
    cols = 20
    num_patches = rows * cols
    num_classes = 28
    
    mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    backbone = Backbone(model=mobilenet)
    patch_embedding = PatchEmbedding(
        backbone=backbone,
        num_patches=num_patches,
        rows=rows,
        cols=cols
    )
    backbone_output_dim = 1000
    pos_embedding = PosEmbedding(
        dim=backbone_output_dim,
        num_patches=num_patches
    )
    patch_embed = PatchEmbedWithPos(
        patch_embedding=patch_embedding,
        pos_embedding=pos_embedding
    )
    
    transformer = TransformerBlock(
        n_block=1,
        nhead=2,
        dim=backbone_output_dim,
        drop_out=0.1,
        use_causal_mask=True
    )
    
    model = ModelTextRecoginizer(
        patchembedwithpos=patch_embed,
        transformer=transformer,
        num_classes=num_classes,
        embedding_dim=backbone_output_dim,
        hidden_classifier=512,
        dropout=0.1
    )
    
    return model


def decode_prediction(logits, debug=False):
    # Logits: [Seq, Batch, Classes] or [Batch, Classes] (if pooled)
    # For CTC, we expect [Seq, Batch, Classes] or similar.
    # Our model returns [Batch, Seq, Classes] if return_sequence=True
    
    # logits shape: [1, 20, 28]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    max_probs, predicted_indices = torch.max(probabilities, dim=-1)
    
    if debug:
        print(f"\n=== DECODE DEBUG ===")
        print(f"Max probs shape: {max_probs.shape}")
        print(f"Max probs: {max_probs.squeeze().tolist()}")
    
    predicted_indices = predicted_indices.squeeze().tolist()
    
    if debug:
        print(f"Raw predicted indices: {predicted_indices}")
    
    # CTC Decoding (Greedy)
    # 1. Collapse repeated characters
    # 2. Remove blanks (index 27)
    
    decoded_indices = []
    prev_idx = -1
    
    for idx in predicted_indices:
        if idx != prev_idx:
            decoded_indices.append(idx)
        prev_idx = idx
    
    if debug:
        print(f"After collapse repeats: {decoded_indices}")
        
    final_indices = [i for i in decoded_indices if i != 27]
    
    if debug:
        print(f"After remove blanks (27): {final_indices}")
    
    char_to_idx = {chr(i): i-ord('a') for i in range(ord('a'), ord('z')+1)}
    char_to_idx[' '] = 26
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    
    if debug:
        print(f"idx_to_char mapping: {idx_to_char}")
    
    predicted_text = ""
    for idx in final_indices:
        if idx in idx_to_char:
            predicted_text += idx_to_char[idx]
        elif debug:
            print(f"WARNING: idx {idx} not in idx_to_char!")
            
    if debug:
        print(f"Final text: '{predicted_text}'")
        print(f"=== END DEBUG ===\n")
            
    return predicted_text

def main():
    checkpoint_path = "./temp/checkpoints_overfit/best_overfit.pth"
    image_path = "./data_test/text_recognizer_data/000013.jpg"
    labels_file = "./data_test/text_recognizer_data/labels.txt"
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Testing image: {image_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = create_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both full checkpoint dict and direct state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load and Preprocess Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Create batch of 1
    # model expects list of images for patch extraction
    # and rows/cols arguments
    
    with torch.no_grad():
        outputs = model([image], rows=1, cols=20, return_sequence=True)
        # outputs shape: [1, 20, 28] (Batch, Seq, Class)
        
        print(f"Output shape: {outputs.shape}")
        print(f"Output min: {outputs.min():.4f}, max: {outputs.max():.4f}")
        
        # Get predicted classes for each position
        predicted_indices = torch.argmax(outputs, dim=-1)
        print(f"Predicted indices: {predicted_indices.squeeze().tolist()}")
        
        # Show top 3 predictions for first few positions
        print(f"\nTop-3 predictions for first 5 positions:")
        for i in range(min(5, outputs.shape[1])):
            top3_probs, top3_indices = torch.topk(torch.softmax(outputs[0, i], dim=-1), k=3)
            print(f"  Pos {i}: {top3_indices.tolist()} (probs: {top3_probs.tolist()})")
        
        prediction = decode_prediction(outputs, debug=True)
        
    print("\n" + "="*40)
    print(f"Prediction: '{prediction}'")
    print("="*40)
    
    # Find Ground Truth
    ground_truth = "Unknown"
    img_filename = os.path.basename(image_path)
    with open(labels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # part[0] might be full path or relative
                if img_filename in parts[0]: 
                    ground_truth = ' '.join(parts[1:])
                    break
                    
    print(f"Ground Truth: '{ground_truth}'")
    print("="*40)

if __name__ == "__main__":
    main()
