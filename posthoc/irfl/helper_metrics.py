import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
import torch.nn.functional as F
import typing
from typing import Literal, List
from src.models import repercent, jointopt
import numpy as np
from einops import rearrange



def test_fwd(x, model, device, M= 3):
    images, texts, text_mask = x["images"], x["texts"], x["pad_masks"]
    
    X = [images.to(device), texts.to(device)]
    X_cross_masks = [None, text_mask.bool().to(device)] 

    if M == 3:
        defs, defs_mask = x["definitions"], x["definitions_mask"]
    
        X.append(defs.to(device))
        X_cross_masks.append(defs_mask.bool().to(device))

    # Forward pass through RePercENT
    outputs = model(X, mask = X_cross_masks)
    
    return outputs

def calc_batch_similarities(outputs, distractors, device, comp_mod= 1):
    """
    Args:
        outputs: (B, D) tensor of model outputs for the correct match
        distractors: (B, K, D) tensor of model outputs for the K distractors
        comp_mod: int, which modality to compute similarities for (1 for captions, 2 for definitions, 3 for both). 
        Note that 2 and 3 are only relevant for the 3-modality setting, for the 2-modality setting.
    Returns:
        pos_sim: (B, 4) tensor of cosine similarities, at position 0 is the similarity for the correct match. The rest is for the distractors.
    """
    # Ensure inputs are on the same device
    # shared_text: [B, D]
    if comp_mod < 3:
        shared_text = outputs['S_view'][:, comp_mod, 0]
        shared_text = F.normalize(shared_text, dim=-1)

        # shared_image_answers: [B, D]
        shared_image_answers = outputs['S_view'][:, 0, comp_mod]
        shared_image_answers = F.normalize(shared_image_answers, dim=-1)

        # shared_image_distractors: [B, K, D]
        shared_image_distractors = distractors
        shared_image_distractors = F.normalize(shared_image_distractors, dim=-1)

        # answer_sim: [B]
        answer_sim = (shared_text * shared_image_answers).sum(dim=-1)


        # distractor_sims: [B, K]
        # einsum computes dot(shared_text[b], shared_image_distractors[b, k]) for all b,k
        distractor_sims = torch.einsum('bd,bkd->bk', shared_text, shared_image_distractors)
    else:
        for i in range(1, 3):
            shared_text = outputs['S_view'][:, i, 0]
            shared_text = F.normalize(shared_text, dim=-1)

            shared_image_answers = outputs['S_view'][:, 0, i]
            shared_image_answers = F.normalize(shared_image_answers, dim=-1)

            shared_image_distractors = distractors[i - 1] # distractors is a list of two tensors in this case
            shared_image_distractors = F.normalize(shared_image_distractors, dim=-1)

            answer_sim_i = (shared_text * shared_image_answers).sum(dim=-1)
            distractor_sims_i = torch.einsum('bd,bkd->bk', shared_text, shared_image_distractors)

            if i == 1:
                answer_sim = answer_sim_i
                distractor_sims = distractor_sims_i
            else:
                # average similarity over the two modalities
                answer_sim = (answer_sim_i + answer_sim) / 2
                distractor_sims = (distractor_sims_i + distractor_sims) / 2


    # combine sims into one tensor: [B, K+1]
    pos_sim = torch.cat([answer_sim.unsqueeze(1), distractor_sims], dim=1)

    return pos_sim



def extract_all_sims(model, test_loader, device, M= 3, comp_mod= 1):
    total_sims = {"overall": [], "metaphor": [], "idiom": [], "simile": []}
    model.eval()
    with torch.inference_mode():
        for batch_idx, out in enumerate(test_loader):
            print(f"Processing batch {batch_idx + 1}/ {len(test_loader)}")
            x = out['x']
            x_aug = out['x_aug']
            temp_b = x['images'].shape[0]
            
            outputs = test_fwd(x, model, device, M= M)
        
            distractors = out['distractors'].to(device)

            B, N, S, D = distractors.shape
            distr_flat = rearrange(distractors, 'b n s d -> (b n) s d')
            if hasattr(model, "disenEncoders"): # This is the RePercENT case
                if comp_mod < 3:
                    out_distractors_flat = model.get_slot(model.disenEncoders[0](distr_flat), 1, f"S_1{comp_mod + 1}")
                    out_distractors = rearrange(out_distractors_flat, '(b n) ... -> b n ...', b=B, n=N)
                else:
                    out_distractors_flat_2 = model.get_slot(model.disenEncoders[0](distr_flat), 1, f"S_12") # image - caption shared
                    out_distractors_flat_3 = model.get_slot(model.disenEncoders[0](distr_flat), 1, f"S_13") # image - definition shared
                    out_distractors_2 = rearrange(out_distractors_flat_2, '(b n) ... -> b n ...', b=B, n=N)
                    out_distractors_3 = rearrange(out_distractors_flat_3, '(b n) ... -> b n ...', b=B, n=N)
                    out_distractors = [out_distractors_2, out_distractors_3]

            elif hasattr(model, "sharedEncoders"): # This is the general JointOpt case
                if comp_mod < 3:
                    out_distractors_flat = model.encode_modality(model.sharedEncoders[f"S_1{comp_mod + 1}"], \
                                model.sharedProjh[f"S_1{comp_mod + 1}"],distr_flat, None)

                    out_distractors = rearrange(out_distractors_flat, '(b n) ... -> b n ...', b=B, n=N)
                else:
                    out_distractors_flat_2 = model.encode_modality(model.sharedEncoders[f"S_12"], \
                                    model.sharedProjh[f"S_12"],distr_flat, None)
                    out_distractors_flat_3 = model.encode_modality(model.sharedEncoders[f"S_13"], \
                                    model.sharedProjh[f"S_13"],distr_flat, None)
                    out_distractors_2 = rearrange(out_distractors_flat_2, '(b n) ... -> b n ...', b=B, n=N)
                    out_distractors_3 = rearrange(out_distractors_flat_3, '(b n) ... -> b n ...', b=B, n=N)
                    out_distractors = [out_distractors_2, out_distractors_3]

            sims = calc_batch_similarities(outputs, out_distractors, device, comp_mod= comp_mod)
            total_sims["overall"] += sims.cpu().numpy().tolist()
            ftypes = out["figurative_type"]
            
            for i, ftype in enumerate(ftypes):
                total_sims[ftype].append(sims[i].cpu().numpy().tolist())
                

    return total_sims



def evaluate_model(model, test_loader, device, M= 3, comp_mod= 1):
    total_sims = extract_all_sims(model, test_loader, device, M= M, comp_mod= comp_mod)

    metrics = {}
    for ftype in total_sims.keys():
        sims = np.array(total_sims[ftype])  # Convert list of sims to a numpy array
        # Calculate metrics for this figurative type
        pos_sim = sims[:, 0]  # Similarity of the correct match
        neg_sims = sims[:, 1:]  # Similarities of the distractors
        max_distractor_sim = neg_sims.max(axis=1)  # Max similarity among distractors
        

        # Mean Reciprocal Rank (MRR)
        ranks = (neg_sims > pos_sim[:, None]).sum(axis=1) + 1  # Rank of the correct match
        mrr = (1 / ranks).mean()

        # Mean margin of the correct match over the most similar distractor
        margin = (pos_sim - max_distractor_sim).mean()

        # Accuracy @1
        acc_at_1 = (pos_sim > max_distractor_sim).mean()

        pair_correct = []

        pair_indices = [(0, 1), (0, 2), (1, 2)]
        for i, j in pair_indices:
            pair_max = np.maximum(neg_sims[:, i], neg_sims[:, j])
            pair_acc = (pos_sim > pair_max).mean()
            pair_correct.append(pair_acc)

        acc_at_2_pairavg = np.mean(pair_correct)

        metrics[ftype] = {
            "acc@1": float(acc_at_1),
            "acc@2_pairavg": float(acc_at_2_pairavg),
            "MRR": float(mrr),
            "margin": float(margin),
        }

    return metrics