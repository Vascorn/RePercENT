import torch
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import open_clip as clip
from src.models.pretrained_encoders.clip_embeddings import get_image
import numpy as np
import math
import torch.nn.functional as F
import pandas as pd
import json, ast
from tqdm import tqdm
from einops import rearrange

# Helper function to load the dataframe rows.
def parse_definition(cell):
    """Return List[str] consistently."""
    if cell is None:
        return []

    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]

    s = str(cell).strip()
    if not s:
        return []

    # try json
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        if isinstance(obj, str):
            s2 = obj.strip()
            # maybe python-list as string
            if s2.startswith("[") and s2.endswith("]"):
                try:
                    obj2 = ast.literal_eval(s2)
                    if isinstance(obj2, list):
                        return [str(x).strip() for x in obj2 if str(x).strip()]
                except Exception:
                    pass
            return [s2] if s2 else []
        return [str(obj).strip()] if str(obj).strip() else []
    except json.JSONDecodeError:
        pass

    # try python literal directly
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass

    return [s]


class CLIP_ft_dataset(torch.utils.data.Dataset):
    def __init__(self, data_df, preprocess_func, tokenizer=None, test_mode=False):
        
        self.preprocess_func = preprocess_func
        self.tokenizer = tokenizer if tokenizer is not None else clip.get_tokenizer("ViT-B-32") # hardcoded for now, can be parameterized later
        self.test_mode = test_mode
        

        # Extract relevant columns from the dataframe
        self.phrases = data_df["phrase"].tolist()
        if self.test_mode:
            self.images, self.distractors, self.definitions, self.joint_inputs, self.fig_types = [], [], [], [], []
            for _, row in tqdm(data_df.iterrows(), total=len(data_df)):
                distractors = [im for im in json.loads(row.distractors)]
                answer = json.loads(row.answer)[0]
                phrase = row.phrase
                definition_list = parse_definition(row.definition)
                definition = '. '.join(definition_list) + '.'
                joint_def_phrase = phrase + '. ' + definition

                self.images.append(answer)
                self.distractors.append(distractors)
                self.definitions.append(definition)
                self.joint_inputs.append(joint_def_phrase)
                self.fig_types.append(row.figurative_type)

            print(f"Test mode enabled. Example distractor filename: {self.distractors[0]}, Example definition: {self.definitions[0]}, Example joint input: {self.joint_inputs[0]}, Example figurative type: {self.fig_types[0]}")
        else:
            self.images = data_df["uuid"].tolist()
            print(f"Initialized CLIP_ft_dataset with images: {len(self.images)} and texts: {len(self.phrases)}")
            print(f"Example image filename: {self.images[0]}, Example phrase: {self.phrases[0]}")
        # tokenize text and preprocess images in advance
        self.prepare_clip_inputs()
        
    def prepare_clip_inputs(self):
        self.preprocess_images = torch.stack([self.preprocess_func(get_image(im)) for im in self.images], dim=0).float()
        self.preprocess_phrases = self.tokenizer(self.phrases)
        if self.test_mode:
            cat_dists = lambda lst: torch.stack([self.preprocess_func(get_image(im)) for im in lst], dim= 0).float() # [num_distractors, 3, 224, 224]
            cat_dists_all = lambda lst_of_lst: torch.stack([cat_dists(lst) for lst in tqdm(lst_of_lst, total=len(lst_of_lst))], dim=0) # [B, num_distractors, 3, 224, 224]
            self.preprocess_distractors = cat_dists_all(self.distractors)
            self.preprocess_definitions = self.tokenizer(self.definitions)
            self.preprocess_joint_inputs = self.tokenizer(self.joint_inputs)
            self.fig_types = self.fig_types

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        print(f"CLIP_ft_dataset with :\nNumber of samples: {len(self.images)}\nTest mode: {self.test_mode}")
        if self.test_mode:
            print(f"Preprocessed image shapers: {self.preprocess_images.shape}, Tokenized phrase shapes: {self.preprocess_phrases.shape}, \n\
                \tPreprocessed distractor shapes: {self.preprocess_distractors.shape}, Tokenized definition shapes: {self.preprocess_definitions.shape}, \n\
                \tTokenized joint input shapes: {self.preprocess_joint_inputs.shape}")
        else:
            print(f"Preprocessed image shapers: {self.preprocess_images.shape}, Tokenized phrase shapes: {self.preprocess_phrases.shape}")
        
        return super().__repr__()

    def __getitem__(self, idx):
        
        img = self.preprocess_images[idx]
        phrase = self.preprocess_phrases[idx]

        if self.test_mode:
            definition = self.preprocess_definitions[idx]
            joint_input = self.preprocess_joint_inputs[idx]
            fig_type = self.fig_types[idx]
            distractor = self.preprocess_distractors[idx]
            return {
                "answer": img,
                "distractor": distractor,
                "phrase": phrase,
                "definition": definition,
                "joint_input": joint_input,
                "fig_type": fig_type
            }
        else:
            return {
                "image": img,
                "phrase": phrase
            }



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
    text_features = F.normalize(text_features, dim=-1)
    image_features = F.normalize(image_features, dim=-1)
    

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


def eval_clip(clip_helper, test_df, device):
    """
    Evaluates clip model on a IRFL test set, computing accuracy for each figurative type and overall.
    Use this version if test set is in dataframe row format (slow).
    Args:
        clip_helper: an instance of CLIPHelper class.
        test_df: a pandas dataframe containing the test set, with columns "uuid", "phrase", "figurative_type", and "definition" (if using definitions).
        device: the device to run the evaluation on (e.g., "cuda" or "cpu".
    """
    results_dict_w_def = {'fig_s_type': [], 'Accuracy': [], "total_correct": 0, "total": 0}
    results_dict_no_def = {'fig_s_type': [], 'Accuracy': [], "total_correct": 0, "total": 0}
    results_dict_def_only = {'fig_s_type': [], 'Accuracy': [], "total_correct": 0, "total": 0}

    for fig_type in test_df['figurative_type'].drop_duplicates():
        results_dict_w_def['fig_s_type'].append(fig_type)
        results_dict_no_def['fig_s_type'].append(fig_type)
        results_dict_def_only['fig_s_type'].append(fig_type)

        # Calculate accuracy for this category
        category_df = test_df[test_df['figurative_type'] == fig_type]

        accuracy, correct, total = clip_helper.compute_accuracy(category_df)
        accuracy_no_def, correct_no_def, total_no_def = clip_helper.compute_accuracy(category_df, use_definitions=False)
        accuracy_def_only, correct_def_only, total_def_only = clip_helper.compute_accuracy(category_df, use_definitions=True, definitions_only=True)
        accuracy_w_def, correct_w_def, total_w_def = clip_helper.compute_accuracy(category_df, use_definitions=True, definitions_only=False)

        results_dict_w_def["total_correct"] += correct_w_def
        results_dict_w_def["total"] += total_w_def
        results_dict_w_def['Accuracy'].append(accuracy_w_def)
        results_dict_no_def['Accuracy'].append(accuracy_no_def)
        results_dict_no_def["total_correct"] += correct_no_def
        results_dict_no_def["total"] += total_no_def
        results_dict_def_only['Accuracy'].append(accuracy_def_only)
        results_dict_def_only["total_correct"] += correct_def_only
        results_dict_def_only["total"] += total_def_only

        print(f"Category: ({fig_type}), Accuracy w/ definitions: {100* accuracy_w_def:.4f} %")
        print(f"Category: ({fig_type}), Accuracy w/o definitions: {100* accuracy_no_def:.4f} %")
        print(f"Category: ({fig_type}), Accuracy def only: {100* accuracy_def_only:.4f} %")

    print(f"Overall Accuracy w/ definitions: {100* results_dict_w_def['total_correct'] / results_dict_w_def['total']:.4f} %")
    print(f"Overall Accuracy w/o definitions: {100* results_dict_no_def['total_correct'] / results_dict_no_def['total']:.4f} %")
    print(f"Overall Accuracy def only: {100* results_dict_def_only['total_correct'] / results_dict_def_only['total']:.4f} %")

    print("Results with definitions:", results_dict_w_def)
    print("Results without definitions:", results_dict_no_def)
    print("Results with definitions only:", results_dict_def_only)            
    return results_dict_w_def, results_dict_no_def, results_dict_def_only


def eval_clip_tensor(clip_helper, test_loader, device):
    """
    Evaluates clip model on a IRFL test set, computing similarity scores for each figurative type and overall.
    This version assumes the test set is already preprocessed, tokenized and provided as a dataloader (fast).
    Args:
    clip_helper: an instance of CLIPHelper class.
    test_loader: a PyTorch DataLoader that yields batches of preprocessed images, tokenized phrases, definitions, joined phrase+definitions, distractors, and figurative types.
    device: the device to run the evaluation on (e.g., "cuda" or "cpu").
     
    """
    fig_cats = ["idiom", "metaphor", "simile"]
    results_dict_w_def = {'fig_s_type': fig_cats, 'Accuracy': [0, 0, 0], "counts": [0, 0, 0], "total_correct": 0, "total": 0}
    results_dict_no_def = {'fig_s_type': fig_cats, 'Accuracy': [0, 0, 0], "counts": [0, 0, 0], "total_correct": 0, "total": 0}
    results_dict_def_only = {'fig_s_type': fig_cats, 'Accuracy': [0, 0, 0], "counts": [0, 0, 0], "total_correct": 0, "total": 0}

    # init label map for masking
    label_map = {"idiom": 0, "metaphor": 1, "simile": 2}

    with torch.inference_mode():
        for batch in test_loader:
            images = batch["answer"].to(device)
            definitions = batch["definition"].to(device) # already tokenized and preprocessed definition inputs
            joint_vectors = batch["joint_input"].to(device) # phrase + definition tokenized input
            phrase_vectors = batch["phrase"].to(device) # phrase tokenized input
            

            distractors = batch["distractor"].to(device)
            figurative_types = torch.tensor(
                [label_map[x] for x in batch["fig_type"]],
                device=device
            )

            idiom_mask = figurative_types == 0
            metaphor_mask = figurative_types == 1
            simile_mask = figurative_types == 2

            image_vectors = clip_helper.get_clip_img_vector(images, normalize= True, preprocessed=True) # already preprocessed image inputs
            phrase_vectors = clip_helper.get_clip_txt_vector(phrase_vectors, normalize= True, preprocessed=True)
            definition_vectors = clip_helper.get_clip_txt_vector(definitions, normalize= True, preprocessed=True)
            joint_vectors = clip_helper.get_clip_txt_vector(joint_vectors, normalize= True, preprocessed=True)

            distractors_flat = rearrange(distractors, 'b n c h w -> (b n) c h w') # flatten distractors to [B*num_distractors, 3, 224, 224]
            distractor_vectors_flat = clip_helper.get_clip_img_vector(distractors_flat, normalize= True, preprocessed=True)
            distractor_vectors = rearrange(distractor_vectors_flat, '(b n) d -> b n d', b= images.size(0)) # reshape back to [B, num_distractors, D]

            # Compute similarities w/o defintions
            pos_sims_no_def = (image_vectors * phrase_vectors).sum(dim= -1)
            neg_sims_no_def = torch.einsum('bd,bkd->bk', phrase_vectors, distractor_vectors)
            answers_no_def = torch.cat([pos_sims_no_def.unsqueeze(1), neg_sims_no_def], dim=1) # [B, 1 + num_distractors]

            # Compute similarities w/ definitions       
            pos_sims_w_defs  = (image_vectors * joint_vectors).sum(dim= -1)    
            neg_sims_w_defs = torch.einsum('bd,bkd->bk', joint_vectors, distractor_vectors)
            answers_w_defs = torch.cat([pos_sims_w_defs.unsqueeze(1), neg_sims_w_defs], dim=1) # [B, 1 + num_distractors]

            # Compute similarities for def only    
            pos_sims_def_only= (image_vectors * definition_vectors).sum(dim= -1)
            neg_sims_def_only = torch.einsum('bd,bkd->bk', definition_vectors, distractor_vectors) # [B, num_distractors]
            answers_def_only = torch.cat([pos_sims_def_only.unsqueeze(1), neg_sims_def_only], dim=1) # [B, 1 + num_distractors]
            

            # Update results dictionaries
            for fig_id, (fig_cat, mask) in enumerate(zip(fig_cats, [idiom_mask, metaphor_mask, simile_mask])):
                if mask.sum() > 0:
                    # Compute correct for this category
                    correct_w_def = torch.argmax(answers_w_defs[mask], dim=1).eq(0).sum().item() # check if the positive image (index 0) has the highest similarity
                    total_w_def = mask.sum().item()
                    
                    correct_no_def = torch.argmax(answers_no_def[mask], dim=1).eq(0).sum().item() # same for no definitions
                    total_no_def = mask.sum().item()

                    correct_def_only = torch.argmax(answers_def_only[mask], dim=1).eq(0).sum().item() # same for defs only
                    total_def_only = mask.sum().item()

                    results_dict_w_def['Accuracy'][fig_id] += correct_w_def
                    results_dict_w_def["total_correct"] += correct_w_def
                    results_dict_w_def["total"] += total_w_def
                    results_dict_w_def['counts'][fig_id] += total_w_def

                    results_dict_no_def["Accuracy"][fig_id] += correct_no_def
                    results_dict_no_def["total_correct"] += correct_no_def
                    results_dict_no_def["total"] += total_no_def
                    results_dict_no_def['counts'][fig_id] += total_no_def

                    results_dict_def_only["Accuracy"][fig_id] += correct_def_only
                    results_dict_def_only["total_correct"] += correct_def_only
                    results_dict_def_only["total"] += total_def_only
                    results_dict_def_only['counts'][fig_id] += total_def_only


        for fig_id in range(3):
            for d in (results_dict_w_def, results_dict_no_def, results_dict_def_only):
                c = d["counts"][fig_id]
                d["Accuracy"][fig_id] = d["Accuracy"][fig_id] / c if c > 0 else 0.0

    return results_dict_w_def, results_dict_no_def, results_dict_def_only