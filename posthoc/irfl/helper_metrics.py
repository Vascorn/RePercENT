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


ComponentName = Literal["shared", "unique", "both"]


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


def select_component(outputs, mod_i, mod_j, component: ComponentName):
    if component == "shared":
        return outputs['S_view'][:, mod_i, mod_j]
    if component == "unique":
        return outputs['U'][:, mod_i, mod_j]
    if component == "both":
        return torch.cat([outputs['S_view'][:, mod_i, mod_j], outputs['U'][:, mod_i, mod_j]], dim=-1)
    raise ValueError(f"Unsupported component: {component}")


def calc_batch_similarities(outputs, distractors, device, comp_mod= 1, component: ComponentName = "shared"):
    """
    Args:
        outputs: (B, D) tensor of model outputs for the correct match
        distractors: (B, K, D) tensor of model outputs for the K distractors
        comp_mod: int, which modality to compute similarities for (1 for captions, 2 for definitions, 3 for both). 
        Note that 2 and 3 are only relevant for the 3-modality setting, for the 2-modality setting.
        component: which component to compare ("shared", "unique", or "both" for shared concatenated with unique).
    Returns:
        pos_sim: (B, 4) tensor of cosine similarities, at position 0 is the similarity for the correct match. The rest is for the distractors.
    """
    # Ensure inputs are on the same device
    if comp_mod < 3:
        query = select_component(outputs, comp_mod, 0, component)
        query = F.normalize(query, dim=-1)

        image_answers = select_component(outputs, 0, comp_mod, component)
        image_answers = F.normalize(image_answers, dim=-1)

        image_distractors = distractors
        image_distractors = F.normalize(image_distractors, dim=-1)

        # answer_sim: [B]
        answer_sim = (query * image_answers).sum(dim=-1)


        # distractor_sims: [B, K]
        # einsum computes dot(query[b], image_distractors[b, k]) for all b,k
        distractor_sims = torch.einsum('bd,bkd->bk', query, image_distractors)
    else:
        for i in range(1, 3):
            query = select_component(outputs, i, 0, component)
            query = F.normalize(query, dim=-1)

            image_answers = select_component(outputs, 0, i, component)
            image_answers = F.normalize(image_answers, dim=-1)

            image_distractors = distractors[i - 1] # distractors is a list of two tensors in this case
            image_distractors = F.normalize(image_distractors, dim=-1)

            answer_sim_i = (query * image_answers).sum(dim=-1)
            distractor_sims_i = torch.einsum('bd,bkd->bk', query, image_distractors)

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


def encode_image_distractors(model, distr_flat, batch_size, num_distractors, comp_mod, component: ComponentName):
    if hasattr(model, "disenEncoders"): # This is the RePercENT case
        image_components = model.disenEncoders[0](distr_flat)
        shared_flat = model.get_slot(image_components, 1, f"S_1{comp_mod + 1}")
        if component == "shared":
            out_distractors_flat = shared_flat
        else:
            unique_flat = model.get_slot(image_components, 1, f"U_1{comp_mod + 1}")
            out_distractors_flat = unique_flat if component == "unique" else torch.cat([shared_flat, unique_flat], dim=-1)

    elif hasattr(model, "sharedEncoders"): # This is the general JointOpt case
        shared_key = f"S_1{comp_mod + 1}"
        shared_flat = model.encode_modality(model.sharedEncoders[shared_key], \
                    model.sharedProjh[shared_key], distr_flat, None)
        if component == "shared":
            out_distractors_flat = shared_flat
        else:
            unique_key = f"U_1{comp_mod + 1}"
            unique_flat = model.encode_modality(model.uniqueEncoders[unique_key], \
                        model.uniqueProjh[unique_key], distr_flat, None)
            out_distractors_flat = unique_flat if component == "unique" else torch.cat([shared_flat, unique_flat], dim=-1)

    else:
        raise ValueError("Expected a RePercENT or JointOpt-style model with component encoders.")

    return rearrange(out_distractors_flat, '(b n) ... -> b n ...', b=batch_size, n=num_distractors)



def extract_all_sims(model, test_loader, device, M= 3, comp_mod= 1, component: ComponentName = "shared"):
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
            if comp_mod < 3:
                out_distractors = encode_image_distractors(model, distr_flat, B, N, comp_mod, component)
            else:
                out_distractors = [
                    encode_image_distractors(model, distr_flat, B, N, i, component)
                    for i in range(1, 3)
                ]

            sims = calc_batch_similarities(outputs, out_distractors, device, comp_mod= comp_mod, component= component)
            total_sims["overall"] += sims.cpu().numpy().tolist()
            ftypes = out["figurative_type"]
            
            for i, ftype in enumerate(ftypes):
                total_sims[ftype].append(sims[i].cpu().numpy().tolist())
                

    return total_sims



def evaluate_model(model, test_loader, device, M= 3, comp_mod= 1, component: ComponentName = "shared"):
    total_sims = extract_all_sims(model, test_loader, device, M= M, comp_mod= comp_mod, component= component)

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
        margin = pos_sim - max_distractor_sim
        margin_mean = margin.mean() # Average margin across all examples
        margin_correct = (margin > 0).mean()  # Proportion of examples where the correct match is more similar than all distractors

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
            "margin": float(margin_mean),
            "margin_correct": float(margin_correct)
        }

    return metrics
