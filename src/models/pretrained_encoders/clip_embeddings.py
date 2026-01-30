import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess_func = clip.load('ViT-B/32', device=device)

total_parameters = sum(p.numel() for p in model.parameters())
for param in model.parameters():
    param.requires_grad = False

@torch.no_grad()
def get_text_token_embeddings(texts):
    tokens = clip.tokenize(texts, truncate=True).to(device)          # [B, T]

    x = model.token_embedding(tokens).type(model.dtype)              # [B, T, d_model]
    x = x + model.positional_embedding.type(model.dtype)             # [B, T, d_model]

    x = x.permute(1, 0, 2)                                           # [T, B, d_model]
    x = model.transformer(x)
    x = x.permute(1, 0, 2)                                           # [B, T, d_model]

    x = model.ln_final(x)                                            # [B, T, d_model]

    mask = tokens != 0 # [B, T]
    return x.float(), tokens, mask                                     # token-level features (pre-projection)

@torch.no_grad()
def get_clip_txt_vector(text, normalize=True):
    cue_clip_txt = clip.tokenize([text]).to(device)
    
    cue_clip_txt_encoded = model.encode_text(cue_clip_txt)
    
    if normalize:
        cue_clip_txt_encoded /= cue_clip_txt_encoded.norm(dim=-1, keepdim=True)
    return cue_clip_txt_encoded


@torch.no_grad()
def get_vit_patch_embeddings(images):
    """
    images: list[PIL.Image] OR a preprocessed tensor [B, 3, 224, 224]
    returns:
      patch_feats: [B, N, width]   (pre-projection)
      cls_feat:    [B, width]      (pre-projection)
    """
    visual = model.visual

    if isinstance(images, torch.Tensor):
        x = images.to(device)
    else:
        x = torch.stack([preprocess_func(get_image(im)) for im in images], dim=0).to(device)

    x = x.type(model.dtype)

    # conv1 patchify
    x = visual.conv1(x)                                               # [B, width, grid, grid]
    B, C, Gh, Gw = x.shape
    x = x.reshape(B, C, Gh * Gw).permute(0, 2, 1)                     # [B, N, width]

    # add CLS token
    cls_t = visual.class_embedding.to(x.dtype)
    cls_t = cls_t + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls_t, x], dim=1)                                    # [B, 1+N, width]

    # add pos + ln_pre
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    # transformer blocks
    x = x.permute(1, 0, 2)                                            # [L, B, width]
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)                                            # [B, 1+N, width]

    # final LN (still pre-projection)
    x = visual.ln_post(x)                                             # [B, 1+N, width]

    cls_feat = x[:, 0, :].float()                                     # [B, width]
    patch_feats = x[:, 1:, :].float()                                 # [B, N, width]
    return patch_feats, cls_feat

@torch.no_grad()
def get_clip_img_vector(img, normalize=True):
    cue_clip_img = preprocess_func(get_image(img)).unsqueeze(0).to(device)
    
    cue_clip_img_encoded = model.encode_image(cue_clip_img)
    
    if normalize:
        cue_clip_img_encoded /= cue_clip_img_encoded.norm(dim=-1, keepdim=True)
    return cue_clip_img_encoded


def get_vectors_similarity(v1, v2):
    similarity = v1.detach().cpu().numpy() @ v2.detach().cpu().numpy().T
    return similarity



def get_phrase_image_similarity_score(phrase, image, definitions, normalize=True):
    """
    Zero-shot CLIP matching score between a phrase and an image.
    This function will return the matching probability of the phrase and its image.
    If `definitions` are provided, it will concatenate the definitions of the phrase. Only idiom instances pass definitions.
    
    phrase: The figurative language phrase.
    image: The image (PIL.Image).
    definitions: List of definitions (strings) corresponding to the phrase. Can be empty list if no definitions are to be used.
    normalize: Whether to normalize the CLIP embeddings before computing similarity. Default is True, to get cosine similarity.
    Returns:
        The similarity score (float).
    
    """
    if definitions:
        definition_prompt = '.'.join(definitions) + '.'
        phrase_plus_definitions = phrase + '.' + definition_prompt
        phrase_clip_txt_encoded = get_clip_txt_vector(phrase_plus_definitions, normalize=normalize)
    else:
        phrase_clip_txt_encoded = get_clip_txt_vector(phrase, normalize=normalize)
    clip_cand_img_encoded = get_clip_img_vector(image, normalize=normalize)
    cand_txt_img_sim = get_vectors_similarity(phrase_clip_txt_encoded, clip_cand_img_encoded).item()
    return cand_txt_img_sim