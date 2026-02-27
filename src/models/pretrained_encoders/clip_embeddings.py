import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import torch
import open_clip as clip
from PIL import Image
import json
image_path_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/irfl/images/")

# Loading images from the folder
def get_image_path_from_folder(image_name, image_folder_path=image_path_folder):
    image_path = image_folder_path + image_name.split(".")[0] + ".jpeg"
    return image_path

def get_image(image_name):
    image_path = get_image_path_from_folder(image_name)
    return Image.open(image_path)


class CLIPHelper:
    def __init__(self, device="cuda", clip_model_name="ViT-B-32", model= None, preprocess_func= None):
        self.device = device
        self.clip_model_name = clip_model_name
        if model is None or preprocess_func is None:
            self.clip_model, _, self.preprocess_func = clip.create_model_and_transforms(clip_model_name, pretrained='openai', device=device)
            self.tokenizer = clip.get_tokenizer(clip_model_name)
        else:
            self.clip_model = model
            self.preprocess_func = preprocess_func
            self.tokenizer = clip.get_tokenizer(clip_model_name)


    # Clip specific helper functions to get text and image embeddings, and compute similarity.
    @torch.no_grad()
    def get_text_token_embeddings(self, texts):
        tokens = self.tokenizer(texts).to(self.device)          # [B, T]

        x = self.clip_model.token_embedding(tokens).type(self.clip_model.dtype)              # [B, T, d_model]
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)             # [B, T, d_model]

        x = x.permute(1, 0, 2)                                           # [T, B, d_model]
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)                                           # [B, T, d_model]

        x = self.clip_model.ln_final(x)                                            # [B, T, d_model]

        mask = tokens != 0 # [B, T]
        return x.float(), tokens, mask                                     # token-level features (pre-projection)


    @torch.no_grad()
    def get_vit_patch_embeddings(self, images):
        """
        images: list[PIL.Image] OR a preprocessed tensor [B, 3, 224, 224]
        returns:
        patch_feats: [B, N, width]   (pre-projection)
        cls_feat:    [B, width]      (pre-projection)
        """
        visual = self.clip_model.visual

        if isinstance(images, torch.Tensor):
            x = images.to(self.device)
        else:
            x = torch.stack([self.preprocess_func(get_image(im)) for im in images], dim=0).to(self.device)

        x = x.type(self.clip_model.dtype)

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
    def get_clip_txt_vector(self, text, normalize=True, preprocessed= False):
        if not preprocessed:
            cue_clip_txt = self.tokenizer([text]).to(self.device)
        else:
            cue_clip_txt = text.to(self.device)
        
        cue_clip_txt_encoded = self.clip_model.encode_text(cue_clip_txt)
        
        if normalize:
            cue_clip_txt_encoded /= cue_clip_txt_encoded.norm(dim=-1, keepdim=True)
        return cue_clip_txt_encoded

    @torch.no_grad()
    def get_clip_img_vector(self, img, normalize=True, preprocessed= False):
        if not preprocessed:
            cue_clip_img = self.preprocess_func(get_image(img)).unsqueeze(0).to(self.device)
        else:
            cue_clip_img = img.to(self.device)
        
        cue_clip_img_encoded = self.clip_model.encode_image(cue_clip_img)
        
        if normalize:
            cue_clip_img_encoded /= cue_clip_img_encoded.norm(dim=-1, keepdim=True)
        return cue_clip_img_encoded

    @staticmethod   
    def get_vectors_similarity(v1, v2):
        similarity = v1.detach().cpu().numpy() @ v2.detach().cpu().numpy().T
        return similarity



    def get_phrase_image_similarity_score(self, phrase, image, definitions, normalize=True):
        """
        Zero-shot CLIP matching score between a phrase and an image.
        This function will return the matching probability of the phrase and its image.
        If `definitions` are provided, it will concatenate the definitions of the phrase. Only idiom instances pass definitions.
        Args:
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
            phrase_clip_txt_encoded = self.get_clip_txt_vector(phrase_plus_definitions, normalize=normalize)
        else:
            phrase_clip_txt_encoded = self.get_clip_txt_vector(phrase, normalize=normalize)
        clip_cand_img_encoded = self.get_clip_img_vector(image, normalize=normalize)
        cand_txt_img_sim = self.get_vectors_similarity(phrase_clip_txt_encoded, clip_cand_img_encoded).item()
        return cand_txt_img_sim


    def compute_accuracy(self, test_df, use_definitions=True, definitions_only=False):
        correct = 0
        total = 0

        for row in test_df.itertuples():
            distractors = json.loads(row.distractors)
            answer = json.loads(row.answer)[0]
            phrase = row.phrase
            definition = json.loads(row.definition)

            scores = []
            for img_id in distractors + [answer]:
                if definitions_only:
                    score = self.get_phrase_image_similarity_score('.'.join(definition) + '.', img_id, "", normalize=True)
                else:
                    score = self.get_phrase_image_similarity_score(phrase, img_id, definition if use_definitions else "", normalize=True)
                scores.append((img_id, score))
            
            # Get the image with the highest score
            predicted_img_id, _ = max(scores, key=lambda x: x[1])
            
            if predicted_img_id == answer:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        return accuracy, correct, total

        