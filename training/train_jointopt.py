import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
from typing import Literal, List
from torch.utils.data import random_split
import wandb
from src.utils.helpers import extract_latents_and_labels, linear_probe, plot_confusion_matrix
from src.models.jointopt_2m import MLP
from src.models.jointopt import GRUEncoder, JointOpt
from src.models.third_party.g_mlp_repo.g_mlp.core import gMLP
import numpy as np
import math

def build_encoders(cfg: dict):
    t = cfg["type"].lower()
    encs = []

    if t in ("mlp", "gru"):
        in_dims= cfg["input_dims"]
        hid_dims= cfg["hidden_dims"]
        lat_dims= cfg["latent_dims"]
        act= cfg.get("activation", "relu")

        proj_hds = None # not needed for MLP or GRU encoders

        for in_d, hds, lat_d in zip(in_dims, hid_dims, lat_dims):
            if t == "mlp":
                enc = MLP(input_dim=in_d, hidden_dims=hds, latent_dim=lat_d, activation=act)
            else:  # gru
                enc = GRUEncoder(
                    input_dim=in_d,
                    hidden_dim=hds[0],
                    latent_dim=lat_d,
                    num_layers=cfg.get("num_layers", 1),
                    bidirectional=cfg.get("bidirectional", False),
                )
            encs.append(enc)

    elif t in ("gmlp"):
        d_models= cfg["d_model"]
        d_ffs= cfg["d_ff"]
        seq_lens= cfg["seq_len"]
        num_layers= cfg["num_layers"]
        proj_h= cfg["proj_h"]

        # define linear projection heads
        proj_hds = [nn.Linear(ph[0], ph[1]) for ph in proj_h]
        proj_needed = any(ph[0] != ph[1] for ph in proj_h)
        proj_hds = proj_hds if proj_needed else None # remove if not needed
        print(f"Projection heads defined for gMLP encoders: {proj_needed}. {'Using identity projections.' if not proj_needed else f'Projection head dimensions: {proj_h}'}")
        for dm, dff,sl, nl in zip(d_models, d_ffs, seq_lens, num_layers):
            encs.append(gMLP(d_model=dm, d_ffn=dff, seq_len=sl, num_layers=nl))
            
    else:
        raise ValueError(f"Unsupported encoder type: {cfg['type']}")

    return encs, proj_hds


def make_model_jointopt(model_config_jointopt: dict) -> nn.Module:
    '''
    Create JointOpt model based on the model configuration.
    Args:
        model_config (dict): Configuration dictionary for the model.
    Returns:
        JointOpt: Instantiated JointOpt model.
    '''

    sharedEncoders, shared_projh  = build_encoders(model_config_jointopt["shared_encoder"])
    
    uniqueEncoders, unique_projh = build_encoders(model_config_jointopt["unique_encoder"])

    print(f"Built {len(sharedEncoders)} shared encoders and {len(uniqueEncoders)} unique encoders for JointOpt model.")
    model = JointOpt(M= model_config_jointopt["M"], 
                    sharedEncoders= sharedEncoders, 
                    uniqueEncoders= uniqueEncoders, 
                    shared_projh= shared_projh,
                    unique_projh= unique_projh,
                    encoder_type= model_config_jointopt["shared_encoder"]["type"].lower(), # We assume the same encoder type for the shared & unique encoders
                    vmfkappa= model_config_jointopt["vmfkappa"])

    return model