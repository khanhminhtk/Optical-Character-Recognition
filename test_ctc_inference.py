import sys
import os
from pathlib import Path
import torch
import cv2
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from infrastructure.ml.models.vit_ctc.model import ModelTextRecoginizer
from infrastructure.ml.models.vit_ctc.pos_patch_embed import PatchEmbedWithPos
from infrastructure.ml.models.vit_ctc.patch_embedding import PatchEmbedding
from infrastructure.ml.models.vit_ctc.position_embedding import PosEmbedding
from infrastructure.ml.models.vit_ctc.backbone import Backbone
from infrastructure.ml.models.vit_ctc.transformer import TransformerBlock
from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights


def create_model(num_classes=27, num_layers=6, num_heads=8, dropout=0.1, hidden_classifier=512):
    rows = 7
    cols = 4
    num_patches = rows * cols
    
    mobilenet = mobilenet_v3_small(weights=None)
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
        n_block=num_layers,
        nhead=num_heads,
        dim=backbone_output_dim,
        drop_out=dropout
    )
    
    model = ModelTextRecoginizer(
        patchembedwithpos=patch_embed,
        transformer=transformer,
        num_classes=num_classes,
        embedding_dim=backbone_output_dim,
        hidden_classifier=hidden_classifier,
        dropout=dropout
    )
    
    return model


def ctc_decode(logits, blank_idx=26):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    pred_indices = torch.argmax(log_probs, dim=-1)
    
    decoded = []
    prev_idx = None
    for idx in pred_indices:
        idx = idx.item()
        if idx != blank_idx and idx != prev_idx:
            decoded.append(idx)
        prev_idx = idx
    
    return decoded


def indices_to_text(indices):
    idx_to_char = {i: chr(i + ord('a')) for i in range(26)}
    idx_to_char[26] = ' '
    
    text = ''.join([idx_to_char.get(idx, '?') for idx in indices])
    return text


def main():
    checkpoint_path = './temp/checkpoints_ctc/checkpoints_ctc/best_model.pt'
    image_path = './data_test/text_recognizer_data/000004.jpg'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("CTC Model Inference Test")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image: {image_path}")
    print(f"Device: {device}")
    print("="*60)
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return
    
    print("\nCreating model...")
    model = create_model()
    model = model.to(device)
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        print(f"  Val Acc: {checkpoint.get('val_accuracy', 0):.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print("\nLoading image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image from {image_path}")
        return
    
    print(f"  Image shape: {image.shape}")
    
    print("\nRunning inference...")
    rows = 7
    cols = 4
    
    with torch.no_grad():
        outputs = model([image], rows, cols, return_sequence=True)
        print(f"  Output shape: {outputs.shape}")
        
        if outputs.dim() == 3:
            seq_len = outputs.size(1)
            outputs_ctc = outputs.permute(1, 0, 2)
        else:
            outputs_ctc = outputs.unsqueeze(0)
            seq_len = outputs_ctc.size(0)
        
        print(f"  Sequence length: {seq_len}")
        
        decoded_indices = ctc_decode(outputs_ctc[:, 0, :], blank_idx=26)
        decoded_text = indices_to_text(decoded_indices)
        
        print("\n" + "="*60)
        print("RESULT")
        print("="*60)
        print(f"Decoded indices: {decoded_indices}")
        print(f"Decoded text: '{decoded_text}'")
        print("="*60)
        
        top5_probs = torch.nn.functional.softmax(outputs_ctc, dim=-1)
        print(f"\nPer-timestep predictions:")
        for t in range(seq_len):
            probs = top5_probs[t, 0, :]
            top5_vals, top5_idx = torch.topk(probs, 5)
            chars = [indices_to_text([i]) for i in top5_idx.cpu().numpy()]
            print(f"  t={t}: {' '.join([f'{c}({p:.2f})' for c, p in zip(chars, top5_vals.cpu().numpy())])}")


if __name__ == "__main__":
    main()
