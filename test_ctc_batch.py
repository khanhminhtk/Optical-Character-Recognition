import sys
import os
from pathlib import Path
import torch
import cv2
import numpy as np
import glob

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from infrastructure.ml.models.vit_ctc.model import ModelTextRecoginizer
from infrastructure.ml.models.vit_ctc.pos_patch_embed import PatchEmbedWithPos
from infrastructure.ml.models.vit_ctc.patch_embedding import PatchEmbedding
from infrastructure.ml.models.vit_ctc.position_embedding import PosEmbedding
from infrastructure.ml.models.vit_ctc.backbone import Backbone
from infrastructure.ml.models.vit_ctc.transformer import TransformerBlock
from torchvision.models.mobilenetv3 import mobilenet_v3_small


def create_model(num_classes=27, num_layers=6, num_heads=8, dropout=0.1, hidden_classifier=512, rows=7, cols=4):
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
    checkpoint_path = './temp/checkpoints_after_custom/checkpoints_ctc/best_model.pth'
    images_dir = './data_test/text_recognizer_data/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("CTC Model Batch Inference Test")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Images dir: {images_dir}")
    print(f"Device: {device}")
    print(f"Config: rows=7, cols=4 (28 patches)")
    print("="*80)
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    all_images = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
    
    if len(all_images) == 0:
        print(f"ERROR: No images found in {images_dir}")
        return
    
    print(f"\nTotal images in dataset: {len(all_images)}")
    
    import random
    random.seed(None)
    image_files = random.sample(all_images, min(5, len(all_images)))
    
    print(f"Selected {len(image_files)} random images for testing\n")
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {images_dir}")
        return
    
    print(f"\nFound {len(image_files)} images")
    
    print("\nCreating model...")
    model = create_model()
    model = model.to(device)
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    rows = 7
    cols = 4
    
    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80)
    
    with torch.no_grad():
        for idx, image_path in enumerate(image_files, 1):
            image = cv2.imread(image_path)
            if image is None:
                print(f"{idx}. {os.path.basename(image_path)}: ERROR - Could not read image")
                continue
            
            try:
                outputs = model([image], rows, cols, return_sequence=True)
            except TypeError:
                out_patch = model.patchembedwithpos([image], rows, cols)
                out_transformer = model.transformer(out_patch)
                outputs = model.classifier(out_transformer)
            
            if outputs.dim() == 3:
                outputs_ctc = outputs.permute(1, 0, 2)
            else:
                outputs_ctc = outputs.unsqueeze(0)
            
            decoded_indices = ctc_decode(outputs_ctc[:, 0, :], blank_idx=26)
            decoded_text = indices_to_text(decoded_indices)
            
            probs = torch.nn.functional.softmax(outputs_ctc[:, 0, :], dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            avg_confidence = max_probs.mean().item()
            
            img_name = os.path.basename(image_path)
            display_text = decoded_text if decoded_text else "(empty)"
            print(f"{idx:2d}. {img_name:15s} | Pred: {display_text:20s} | Conf: {avg_confidence:.2f}")
    
    print("="*80)


if __name__ == "__main__":
    main()
