import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.functional as F
import typing
from typing import Literal, List
from src.models import repercent, jointopt
from src.models.repercent import RePercENT
from src.utils.helpers import ProbeEvaluator, extract_latents_and_labels, linear_probe, non_linear_probe, regression_probe, plot_confusion_matrix, plot_pairwise_confusion_matrices
from training.train_repercent import make_model
from training.train_jointopt_2m import make_model_jointopt
from posthoc.synthetic.helper_metrics import linear_probe_disentanglement_metric
import yaml
import argparse
import wandb
from src.utils.helpers import set_seed
import numpy as np
import matplotlib.pyplot as plt
from training.main import aggregate_and_log
from torch.profiler import profile, ProfilerActivity


def calculate_flops(model, loader, device):
    model.eval()

    # Grab one batch and calculate FLOPs using PyTorch profiler
    X_batch, _, _, _, _ = next(iter(loader))
    X_batch = [x.to(device) for x in X_batch]

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
        profile_memory=False,
    ) as prof:
        with torch.no_grad():
            model(X_batch, mask = [None for _ in range(len(X_batch))])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Aggregate FLOPs over all ops
    total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
    print(f"Estimated FLOPs: {total_flops / 1e6:.2f} MFLOPs")
    print(prof.key_averages().table(sort_by="flops", row_limit=15))

    return total_flops


def main():
    
    parser = argparse.ArgumentParser(description="Post hoc evaluation of trained models on synthetic data")
    parser.add_argument('--datasets_path', type=str, default="../../data/irfl/datasets/", help='Path to the directory containing the IRFL dataset tensors wrt to this script')
    parser.add_argument('--model_type', type=str, choices= ['repercent', 'jointopt'], default='jointopt', help='Type of model to train, for now only repercent is implemented')
    parser.add_argument('--enc_type', type=str, choices= ['gMLP', 'MLP', 'GRU'], default= 'gMLP', help= "The different baseline encoders, if <model_type> is jointopt. This argument \
                                                                                                        is inactive, if model type is repercent")
    parser.add_argument('--M', type=int, default= 4, help='Number of modalities in the synthetic setup.')
    
    # Define number of splits and seeds - these should exactly match the training ones for reproducibility
    parser.add_argument('--k1', type=int, default= 3, help='Number of different train/test splits')
    parser.add_argument('--k2', type=int, default= 2, help='Number of training seeds per split')
    parser.add_argument('--base_seed', type=int, default=2, help='Base seed for reproducibility')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    M = args.M # number of modalities
    
    # Loading configurations for data, model, and training
    print("Loading configurations...")
    data_config_path = os.path.join(script_dir, "../..", "configs", "data", f"synthetic_data_{M}m.yaml")
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    model_config_path = os.path.join(script_dir, "../..", "configs", "model", f"{args.model_type}_{M}m.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    training_config_path = os.path.join(script_dir, "../..", "configs", "training", f"train_synthetic_{M}m.yaml")
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)

    analysis_config_path = os.path.join(script_dir, "../..", "configs", "posthoc_analysis", f"synthetic_{M}m.yaml")
    with open(analysis_config_path, 'r') as f:
        analysis_config = yaml.safe_load(f)


    if args.model_type == "jointopt" and (args.enc_type.lower() != model_config["shared_encoder"]["type"].lower()):
        raise ValueError(f"Encoder type argument does not match the encoder type specified in the model config. Please check your arguments and configs. \
                        Argument encoder type: {args.enc_type}, Model config encoder type: {model_config['shared_encoder']['type']}")

    data_path = os.path.join(script_dir, "../..", "data", "repercent_synthetic", "dataset24") # change accordingly
    all_metrics_summary = []
    for split_idx in range(args.k1):
        # load data-split
        split_path = os.path.join(data_path, f"data_split_{split_idx}.pt")
        dataset_split = torch.load(split_path, weights_only= False)
        
        train_dataset, test_dataset = dataset_split["train_dataset"], dataset_split["test_dataset"]

        for seed_idx in range(args.k2):
            # set seed for reproducibility
            seed = args.base_seed + split_idx * args.k2 + seed_idx
            set_seed(seed)
            generator = torch.Generator().manual_seed(seed)

            train_loader = DataLoader(train_dataset, batch_size= training_config["training"]["batch_size"], generator=generator)
            test_loader = DataLoader(test_dataset, batch_size= training_config["training"]["batch_size"], generator=generator)
            
            
    
            # model creation based on model_type
            if args.model_type == 'jointopt':
                model = make_model_jointopt(model_config).to(device)
                # load model state dictionary
                model_state_dict = analysis_config[args.model_type][args.enc_type.lower()]["checkpoints"][split_idx * args.k2 + seed_idx]
            else:
                disenEncoders = [make_model(model_config, data_config, modality=m + 1, M=data_config["create_data"]["M"]) for m in range(data_config["create_data"]["M"])]
                model = RePercENT(M=data_config["create_data"]["M"],
                                disenEncoder=disenEncoders,
                                disen_mapping=model_config["repercent"]["disen_mapping"]).to(device)

                model_state_dict = analysis_config[args.model_type]["checkpoints"][split_idx * args.k2 + seed_idx]

            
            temp_state_dict = torch.load(os.path.join(script_dir, "../..", model_state_dict), map_location=device)
            model.load_state_dict(temp_state_dict['model_state_dict'])


            train_data_dict = extract_latents_and_labels(model, train_loader, device)
            test_data_dict = extract_latents_and_labels(model, test_loader, device)

            components = list(train_data_dict['Labels_U'].keys()) + list(train_data_dict['Labels_S'].keys())

            evaluator = ProbeEvaluator(linear_probe= linear_probe, regression_probe= regression_probe)
            evaluator.set_data(train_data_dict= train_data_dict, val_data_dict= test_data_dict, M= M)
            
            linear_results = evaluator.calculate_linear_probe()
            reg_results = evaluator.calculate_reg_probe()

            metrics_summary = evaluator.mean_metrics(linear_results, reg_results, M= M)

            all_metrics_summary.append({
                "split_idx": split_idx,
                "seed_idx": seed_idx,
                "split_seed": split_idx,
                "train_seed": seed,
                "metrics": metrics_summary
            })
            
            
            print(f"Evaluation of split: {split_idx} and seed: {seed_idx} complete!")
    

    run = wandb.init(project= "posthoc_" + model_config["wandb"]["project"], 
                    name= f"aggregate_{args.model_type}" if args.model_type == "repercent" else f"aggregate_{model_config['shared_encoder']['type']}", 
                    reinit=True, config={"k1": args.k1, "k2": args.k2, "base_seed": args.base_seed, "model_type": args.model_type})

    aggregate_and_log(all_metrics_summary) 
    # Log model parameters and flops as well
    model_params = sum(p.numel() for p in model.parameters())
    model_flops = calculate_flops(model, test_loader, device)
    
    wandb.log({"model_params": model_params, "model_flops": model_flops})
    wandb.finish()

if __name__ == "__main__":
    main()