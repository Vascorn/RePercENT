import torch
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import clip
from src.models.pretrained_encoders.clip_embeddings import get_image
import numpy as np
import math
import torch.nn.functional as F

class CLIP_ft_dataset(torch.utils.data.Dataset):
    def __init__(self, train_df, preprocess_func):
        self.train_images = train_df["uuid"].tolist()
        self.train_phrases = train_df["phrase"].tolist()
        self.preprocess_func = preprocess_func
        self.tokenizer = clip.tokenize

        # tokenize text and preprocess images in advance
        self.prepare_clip_inputs()
        
    def prepare_clip_inputs(self):
        self.preprocess_images = torch.tensor([self.preprocess_func(get_image(im)) for im in self.train_images], dtype=torch.float32)
        self.preprocess_phrases = torch.tensor(self.tokenizer(self.train_phrases, truncate=True), dtype=torch.float32)

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        
        img = self.preprocess_images[idx]
        phrase = self.preprocess_phrases[idx]

        return img, phrase



def get_clip_encoded_batch(model, tokens_batch, image_batch):
    """
    tokens_batch: [B, T] tokenized and preprocessed text batch
    image_batch: [B, 3, 224, 224] preprocessed image batch
    returns:
    text_features: [B, D] normalized
    image_features: [B, D] normalized
    """
    
    text_features = model.encode_text(tokens_batch)
    image_features = model.encode_image(image_batch)

    # normalize features across the feature dimension
    text_features /= text_features.norm(dim=-1, keepdim=True)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    return text_features, image_features
    


def get_clip_logit_scale_exp(model):
    logit_scale = model.logit_scale.exp()
    return logit_scale



def cosine_with_warmup(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    # cosine decay to 0
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def clip_symmetric_contrastive_loss(image_features, text_features, logit_scale_exp):
    # Compute cosine similarity and scale by logit_scale_exp
    logits_per_image = logit_scale_exp * image_features @ text_features.t()  # [B, B]
    logits_per_text = logit_scale_exp * text_features @ image_features.t()  # [B, B]

    # Ground truth labels (diagonal)
    batch_size = image_features.size(0)
    labels = torch.arange(batch_size).to(image_features.device)

    # Cross-entropy loss for both directions
    loss_img_to_txt = F.cross_entropy(logits_per_image, labels)
    loss_txt_to_img = F.cross_entropy(logits_per_text, labels)

    return (loss_img_to_txt + loss_txt_to_img) / 2.0