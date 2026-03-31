# Fine-tune pretrained Honeybee models per cancer type
#NOTE: NOT FULLY IMPLEMENTED
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time

import numpy as np
import torch
import yaml
import wandb
from torch.utils.data import DataLoader, Subset

from src.models.repercent import RePercENT, DisenLoss
from src.utils.helpers import set_seed
from training.log_data import log_model_details
from training.main_irfl import aggregate_and_log
from training.train_honeybee import train
from training.train_jointopt_2m import make_model_jointopt
from training.train_repercent import make_dataloaders, make_model


def _get_cancer_type_labels(dataset):
    return [dataset[idx]["cancer_type"] for idx in range(len(dataset))]


def _filter_dataset_by_cancer_type(dataset, cancer_type):
    indices = [idx for idx in range(len(dataset)) if dataset[idx]["cancer_type"] == cancer_type]
    return Subset(dataset, indices)


def _random_split_dataset(dataset, val_fraction, seed):
    if len(dataset) < 2:
        raise ValueError("Need at least 2 samples to create a train/validation split.")

    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_val = max(1, int(round(len(indices) * val_fraction)))
    n_val = min(n_val, len(indices) - 1)

    val_indices = indices[:n_val].tolist()
    train_indices = indices[n_val:].tolist()
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _build_model(model_type, model_config, data_config, device):
    match model_type:
        case "repercent":
            disen_encoders = [
                make_model(model_config, data_config, modality=m + 1, M=data_config["create_data"]["M"])
                for m in range(data_config["create_data"]["M"])
            ]
            return RePercENT(
                M=data_config["create_data"]["M"],
                disenEncoder=disen_encoders,
                disen_mapping=model_config["repercent"]["disen_mapping"],
                vmfkappa=model_config["repercent"].get("vmfkappa"),
            ).to(device)
        case "gmlp" | "gru":
            return make_model_jointopt(model_config).to(device)
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained Honeybee models per cancer type")
    parser.add_argument('--datasets_path', type=str, default="../data/honeybee/datasets/", help='Path to the directory containing the Honeybee dataset tensors wrt to this script')
    parser.add_argument('--cancer_type', type=str, required=True, help='Cancer type to fine-tune on, e.g. TCGA-BRCA')
    parser.add_argument('--wsi_embedding_mode', type=str, choices=['slide', 'patch'], default='slide', help='How to handle WSI embeddings: "slide" for pooling to slide-level, "patch" for keeping patch-level with padding')
    parser.add_argument('--model_type', type=str, choices=['repercent', 'gmlp', 'gru'], default='repercent', help='Type of pretrained model to fine-tune')
    parser.add_argument('--split_seed', type=int, default=42, help='Seed of the precomputed train/test split to load')
    parser.add_argument('--base_seed', type=int, default=2, help='Base seed used to align fine-tuning with the pretrained checkpoints')
    parser.add_argument('--add_val_set', type=bool, default=False, help='Whether to create a validation split from the cancer-type-specific training subset')
    parser.add_argument('--evaluate_final_model', type=bool, default=True, help='Whether to evaluate the best fine-tuned checkpoint on the cancer-type-specific test subset')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    print("Loading configurations...")
    data_config_path = os.path.join(script_dir, "..", "configs", "data", "honeybee_data.yaml")
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    model_config_path = os.path.join(script_dir, "..", "configs", "model", f"{args.model_type}_honeybee.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    training_config_path = os.path.join(script_dir, "..", "configs", "training", "ft_honeybee.yaml")
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)

    analysis_config_path = os.path.join(script_dir, "..", "configs", "posthoc_analysis", "honeybee.yaml")
    with open(analysis_config_path, 'r') as f:
        analysis_config = yaml.safe_load(f)

    checkpoints = analysis_config[args.model_type]['checkpoints']
    n_seeds = analysis_config['hyperparameters']['n_seeds']
    assert n_seeds == len(checkpoints), (
        f"Number of seeds in hyperparameters ({n_seeds}) does not match checkpoints ({len(checkpoints)})."
    )

    dataset_split = torch.load(
        os.path.join(script_dir, args.datasets_path, f"dataset_01_{args.wsi_embedding_mode}_split_{args.split_seed}.pt"),
        weights_only=False,
    )
    full_train_dataset = dataset_split['train']
    test_dataset = dataset_split['test']

    train_dataset = _filter_dataset_by_cancer_type(full_train_dataset, args.cancer_type)
    

    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found for cancer type {args.cancer_type}.")
    if len(test_dataset) == 0:
        raise ValueError(f"No test samples found for cancer type {args.cancer_type}.")

    print(f"Fine-tuning cancer type {args.cancer_type}: {len(train_dataset)} train samples, {len(test_dataset)} test samples")

    cancer_type_slug = args.cancer_type.replace('/', '-').replace(' ', '_')
    group_name = time.strftime("%Y-%m-%d_%H-%M-%S") + f"_Honeybee_{args.model_type}_{cancer_type_slug}_ft"

    all_final_metrics = []
    print(f"Starting fine-tuning runs... n_seeds: {n_seeds}, base_seed: {args.base_seed}, split_seed: {args.split_seed}")

    for seed_idx, checkpoint_path in enumerate(checkpoints):
        train_seed = args.base_seed + seed_idx
        set_seed(train_seed)
        g = torch.Generator().manual_seed(train_seed)

        if args.add_val_set:
            temp_train_dataset, temp_val_dataset = _random_split_dataset(
                train_dataset,
                val_fraction=0.2,
                seed=args.split_seed + seed_idx + 1,
            )
            train_loader, val_loader = make_dataloaders(
                temp_train_dataset,
                temp_val_dataset,
                batch_size=training_config["training"]["batch_size"],
                generator=g,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=training_config["training"]["batch_size"],
                shuffle=True,
                generator=g,
            )
            val_loader = None

        test_loader = DataLoader(
            test_dataset,
            batch_size=training_config["training"]["batch_size"],
            shuffle=False,
        )

        # run = wandb.init(
        #     project=data_config["wandb"]["project"],
        #     group=group_name,
        #     name=f"{group_name}_seed_{seed_idx}",
        #     config={
        #         "model_type": args.model_type,
        #         "cancer_type": args.cancer_type,
        #         "split_seed": args.split_seed,
        #         "train_seed": train_seed,
        #         "pretrained_checkpoint": checkpoint_path,
        #         "wsi_embedding_mode": args.wsi_embedding_mode,
        #         "n_pretrained_seeds": n_seeds,
        #     },
        # )

        # log_model_details(
        #     run,
        #     model_name=args.model_type,
        #     data_config=data_config_path,
        #     model_config=model_config_path,
        #     training_config=training_config_path,
        # )

        model = _build_model(args.model_type, model_config, data_config, device)
        pretrained_checkpoint = torch.load(os.path.join(project_root, checkpoint_path), map_location=device)
        model.load_state_dict(pretrained_checkpoint['model_state_dict'])
        model.to(device)

        disen_loss = DisenLoss(
            alpha=training_config["disen_loss"]["alpha"],
            lmd=training_config["disen_loss"]["lmd"],
            lmd_start_value=training_config["disen_loss"]["lmd_start_value"],
            lmd_end_value=training_config["disen_loss"]["lmd_end_value"],
            lmd_n_iterations=training_config["disen_loss"]["lmd_n_iterations"],
            lmd_start_iteration=training_config["disen_loss"]["lmd_start_iteration"],
            ortho_norm=training_config["disen_loss"]["ortho_norm"],
            M=data_config["create_data"]["M"],
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config["optimizer"]["lr"],
            weight_decay=training_config["optimizer"]["weight_decay"],
        )

    

if __name__ == "__main__":
    main()
