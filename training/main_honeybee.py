import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from typing import Literal, List
from torch.utils.data import DataLoader, Subset
import torch.functional as F
from src.models.perceiver import Perceiver
from src.models.repercent import DisenEncoder, RePercENT, DisenLoss
from training.train_repercent import make_dataloaders, make_model
from training.train_honeybee import train
from training.train_jointopt_2m import make_model_jointopt
from training.log_data import log_model_details, log_model_checkpoint
import math
import numpy as np
import yaml
import argparse
import time
from src.utils.helpers import set_seed
from src.utils.honeybee_dataset import MultimodalTCGA

import wandb
import random
from sklearn.model_selection import train_test_split
from training.main_irfl import aggregate_and_log


DEFAULT_FILTER_CANCER_TYPES = [
    'TCGA-BRCA',
    'TCGA-COAD',
    'TCGA-GBM',
    'TCGA-HNSC',
    'TCGA-KIRC',
    'TCGA-LGG',
    'TCGA-LUAD',
    'TCGA-LUSC',
    'TCGA-OV',
    'TCGA-PRAD',
]


def _get_cancer_type_labels(dataset):
    return [dataset[idx]["cancer_type"] for idx in range(len(dataset))]


def _parse_filter_cancer_types(filter_cancer_types):
    if filter_cancer_types is None:
        return None

    cancer_types = []
    for item in filter_cancer_types:
        cancer_types.extend(ct.strip() for ct in item.split(",") if ct.strip())
    return cancer_types or None


def _format_filter_cancer_types(cancer_types):
    if cancer_types is None:
        return "all"
    return ",".join(cancer_types)


def _filter_dataset_by_cancer_types(dataset, cancer_types):
    if cancer_types is None:
        return dataset

    cancer_type_set = set(cancer_types)
    indices = [
        idx
        for idx in range(len(dataset))
        if dataset[idx]["cancer_type"] in cancer_type_set
    ]
    return Subset(dataset, indices)


def stratified_split_dataset(dataset, test_size, seed):
    indices = np.arange(len(dataset))
    labels = np.asarray(_get_cancer_type_labels(dataset))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return Subset(dataset, train_indices.tolist()), Subset(dataset, test_indices.tolist())



def main():
    parser = argparse.ArgumentParser(description="Train RePercENT model on the Honeybee dataset")
    parser.add_argument('--datasets_path', type=str, default="../data/honeybee/datasets/", help='Path to the directory containing the Honeybee dataset tensors wrt to this script')
    
    parser.add_argument('--load_test_split', type=bool, default=True, help='Whether to load a pre-split dataset with fixed train/test split by cancer type. If False, a new random split will be created.')
    parser.add_argument('--save_test_split', type=bool, default=False, help='Whether to save the created train/test split for reproducibility. Only relevant if --load_test_split is False.')
    parser.add_argument('--split_index', type=int, default=0, help='Index for the train/test split to load or save. Only relevant if --load_test_split is True or --save_test_split is True.')
    parser.add_argument('--wsi_embedding_mode', type=str, choices=['slide', 'patch'], default='slide', help='How to handle WSI embeddings: "slide" for pooling to slide-level, "patch" for keeping patch-level with padding')
    parser.add_argument('--model_type', type=str, choices=['repercent', 'gmlp', 'gru'], default='repercent', help='Type of model to train, for now only repercent is implemented')
    parser.add_argument('--filter_cancer_types', nargs='+', default=DEFAULT_FILTER_CANCER_TYPES, help='Optional cancer types to keep, e.g. --filter_cancer_types TCGA-BRCA TCGA-LUAD or TCGA-BRCA,TCGA-LUAD. If omitted, DEFAUTL_FILTER_CANCER_TYPES will be used.')
    # Define number of splits and seeds
    parser.add_argument('--n_seeds', type=int, default= 5, help='Number of seeds per split for model initialization and training')
    parser.add_argument('--base_seed', type=int, default=2, help='Base seed for model initialization and training reproducibility')
    parser.add_argument('--split_seed', type=int, default=42, help='Seed used only for the reproducible stratified train/test split')
    parser.add_argument('--add_val_set', type=bool, default=False, help= 'Whether to create a validation set from the training data for monitoring validation loss. If not set, the model will be trained and evaluated only on the test set.')
    parser.add_argument('--evaluate_final_model', type=bool, default=True, help='Whether to run a final evaluation of the best model checkpoint on the test set after training. If not set, only validation metrics will be logged during training.')
    args = parser.parse_args()
    filter_cancer_types = _parse_filter_cancer_types(args.filter_cancer_types)
    filter_cancer_types_label = _format_filter_cancer_types(filter_cancer_types)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    # Loading configurations for data, model, and training
    print("Loading configurations...")
    data_config_path = os.path.join(script_dir, "..", "configs", "data", f"honeybee_data.yaml")
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    model_config_path = os.path.join(script_dir, "..", "configs", "model", f"{args.model_type}_honeybee.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    training_config_path = os.path.join(script_dir, "..", "configs", "training", f"train_honeybee.yaml")
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    

    # Load the full dataset once and create a fixed stratified split by cancer type.
    if args.load_test_split:
        dataset_split = torch.load(os.path.join(script_dir, args.datasets_path, f"dataset_01_{args.wsi_embedding_mode}_split_{args.split_seed}.pt"), weights_only=False)
        train_dataset = dataset_split['train']
        test_dataset = dataset_split['test']
        train_dataset = _filter_dataset_by_cancer_types(train_dataset, filter_cancer_types)
        test_dataset = _filter_dataset_by_cancer_types(test_dataset, filter_cancer_types)
    else:
        load_path = os.path.join(script_dir, args.datasets_path, f"dataset_01_{args.wsi_embedding_mode}.pt")
        
        print(f"Loading complete TCGA dataset from path {load_path}...")
        dataset = torch.load(load_path, weights_only=False)
        dataset = _filter_dataset_by_cancer_types(dataset, filter_cancer_types)
        train_dataset, test_dataset = stratified_split_dataset(
            dataset,
            test_size=training_config["training"]["test_size"],
            seed=args.base_seed,
        )
        if args.save_test_split:
            save_path = os.path.join(script_dir, args.datasets_path, f"dataset_01_{args.wsi_embedding_mode}_split_{args.split_seed}.pt")
            torch.save({'train': train_dataset, 'test': test_dataset}, save_path)
            print(f"Saved train/test split dataset to path {save_path}")
    
    if filter_cancer_types is not None:
        if len(train_dataset) == 0:
            raise ValueError(f"No training samples found for cancer types: {filter_cancer_types}.")
        if len(test_dataset) == 0:
            raise ValueError(f"No test samples found for cancer types: {filter_cancer_types}.")
        print(
            f"Filtered cancer types {filter_cancer_types}: "
            f"{len(train_dataset)} train samples, {len(test_dataset)} test samples"
        )
    
    
    group_name = time.strftime("%Y-%m-%d_%H-%M-%S") + f"_Honeybee_{args.model_type}_seeds_{args.n_seeds}"
    # Initialize list to store final metrics across all runs
    all_final_metrics = []
    print(f"Starting training runs... n_seeds: {args.n_seeds}, base_seed: {args.base_seed}, split_seed: {args.split_seed}")
    # Outer loop over different training seeds. The train/test split stays fixed.
    for seed_idx in range(args.n_seeds):
        train_seed = args.base_seed + seed_idx
        set_seed(train_seed)

        g = torch.Generator().manual_seed(train_seed)

        if args.add_val_set:
            temp_train_dataset, temp_val_dataset = stratified_split_dataset(
                train_dataset,
                test_size=0.2,
                seed=args.split_seed + seed_idx + 1,
            )
            train_loader, val_loader = make_dataloaders(
                temp_train_dataset,
                temp_val_dataset,
                batch_size=training_config["training"]["batch_size"],
                generator=g,
            )
        else:
            train_loader = DataLoader(train_dataset, batch_size=training_config["training"]["batch_size"], shuffle=True, generator=g)
            val_loader = None

        test_loader = DataLoader(test_dataset, batch_size=training_config["training"]["batch_size"], shuffle=False)


        # Initialize wandb run and log hyperparameters
        run = wandb.init(
            project=data_config["wandb"]["project"],
            group=group_name,
            name=f"{group_name}_seed_{seed_idx}",
            config={
                "n_seeds": args.n_seeds, "base_seed": args.base_seed, "split_seed": args.split_seed,
                "model_type": args.model_type,
                "filter_cancer_types": filter_cancer_types,
            }
        )
        

        log_model_details(run, model_name=args.model_type, data_config=data_config_path, model_config=model_config_path, training_config=training_config_path)

        # model creation
        if args.model_type == "repercent":
            disenEncoders = [make_model(model_config, data_config, modality=m + 1, M=data_config["create_data"]["M"]) for m in range(data_config["create_data"]["M"])]
            model = RePercENT(M=data_config["create_data"]["M"],
                            disenEncoder=disenEncoders,
                            disen_mapping=model_config["repercent"]["disen_mapping"],
                            vmfkappa=model_config["repercent"]["vmfkappa"]).to(device)
        else:
            model = make_model_jointopt(model_config).to(device)
            
        

        disen_loss = DisenLoss(alpha=training_config["disen_loss"]["alpha"],
                                beta=training_config["disen_loss"]["beta"],
                                    lmd=training_config["disen_loss"]["lmd"],
                                    lmd_start_value=training_config["disen_loss"]["lmd_start_value"],
                                    lmd_end_value=training_config["disen_loss"]["lmd_end_value"],
                                    lmd_n_iterations=training_config["disen_loss"]["lmd_n_iterations"],
                                    lmd_start_iteration=training_config["disen_loss"]["lmd_start_iteration"],
                                    ortho_norm=training_config["disen_loss"]["ortho_norm"],
                                    M=data_config["create_data"]["M"])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config["optimizer"]["lr"],
            weight_decay=training_config["optimizer"]["weight_decay"]
        )

        # run key for logging
        run_key = f"seed{seed_idx}"

        # Logging identifiers
        wandb.log({
            "meta/seed_idx": seed_idx,
            "meta/train_seed": train_seed,
            "meta/split_seed": args.split_seed,
            "meta/filter_cancer_types": filter_cancer_types_label,
        })

        # TRAIN
        final_metrics = train(
            train_loader,
            test_loader,
            model,
            optimizer,
            disen_loss,
            training_config["training"]["n_epochs"],
            device,
            val_loader=val_loader,
            checkpoint_dir=os.path.join(script_dir, '..', 'checkpoints', 'honeybee', run.name, run_key),
            include_reverse_shared_pairwise=training_config["training"].get("include_reverse_shared_pairwise", True),
            evaluate_final_model=args.evaluate_final_model
        )

        # Store + log final snapshot table
        all_final_metrics.append({
            "seed_idx": seed_idx,
            "metrics": final_metrics,
        })
        wandb.finish()

    if args.evaluate_final_model:
        print("All runs completed. Logging average values to wandb...")
        # global summary run
        run = wandb.init(project= data_config["wandb"]["project"], 
                        group=group_name, name=f"aggregate_{args.model_type}_{'w_val_set' if args.add_val_set else 'no_val_set'}", 
                        reinit=True, config={"n_seeds": args.n_seeds, "base_seed": args.base_seed, "split_seed": args.split_seed, "model_type": args.model_type, "filter_cancer_types": filter_cancer_types})
        
        aggregate_and_log(all_final_metrics)
        wandb.finish()

if __name__ == "__main__":
    main()
