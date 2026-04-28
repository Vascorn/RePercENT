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
from src.utils.irfl_dataset import make_dataset
from posthoc.irfl.helper_metrics import evaluate_model
from training.train_repercent import make_dataloaders, make_model
from training.train_jointopt_2m import make_model_jointopt
import yaml
import argparse
import wandb
from src.utils.helpers import set_seed
import numpy as np





def main():
    
    parser = argparse.ArgumentParser(description="Train RePercENT model on the IRFL dataset")
    parser.add_argument('--datasets_path', type=str, default="../../data/irfl/datasets/", help='Path to the directory containing the IRFL dataset tensors wrt to this script')
    parser.add_argument('--model_type', type=str, choices=['repercent', 'gmlp', 'gru'], default='repercent', help='Type of model to train, for now only repercent is implemented')
    parser.add_argument('--comp_mod', type=int, choices=[1, 2, 3], default= 1, help='Which modality to compute similarities for (1 for captions, 2 for definitions, 3 for adding \
                                                                                    the similarities between images- captions and images - definitions and then comparing the metrics). \
                                                                                    Note that 2 and 3 is only relevant for the 3-modality setting')
    parser.add_argument('--component', type=str, choices=['shared', 'unique', 'both'], default='both',
                        help='Which component to assess (shared, unique, or both for shared concatenated with unique).')
    # Define number of splits and seeds
    parser.add_argument('--base_seed', type=int, default= 2, help='Base seed for reproducibility')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    M = 3 # number of modalities, for the IRFL: M = 2 -> images + captions, M = 3 -> images + captions + definitions
    
    # Loading configurations for data, model, and training
    print("Loading configurations...")
    data_config_path = os.path.join(script_dir, "../..", "configs", "data", f"irfl_data_{M}m.yaml")
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    model_config_path = os.path.join(script_dir, "../..", "configs", "model", f"{args.model_type}_irfl_{M}m.yaml")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    analysis_config_path = os.path.join(script_dir, "../..", "configs", "posthoc_analysis", f"irfl_{M}m.yaml")
    with open(analysis_config_path, 'r') as f:
        analysis_config = yaml.safe_load(f)
    
    # seed check
    n_seeds = analysis_config['hyperparameters']['n_seeds']
    assert n_seeds == len(analysis_config[args.model_type]['checkpoints']), f"Number of seeds in hyperparameters ({n_seeds}) does not match number of checkpoints specified for {args.model_type} ({len(analysis_config[args.model_type]['checkpoints'])})"

    # Load the *full dataset once*
    print("Loading datasets...")
    
    total_test_data = torch.load(os.path.join(script_dir, args.datasets_path, 'IRFL_test_tensors_2.pt'), map_location="cpu")

    total_test_data_aug = torch.load(os.path.join(script_dir, args.datasets_path, 'IRFL_test_tensors_aug_2.pt'), map_location="cpu")
    test_dataset, test_data_dict = make_dataset(total_data= total_test_data | total_test_data_aug, num_modalities= data_config["create_data"]["M"], data_type='test', include_original=True)

    print(f"Analysis config: {analysis_config[args.model_type]['checkpoints']}")

    # define project root for loading checkpoints
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # init device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init results storage
    complete_report = {f"seed_{i}": {} for i in range(args.base_seed, args.base_seed + n_seeds)}
    complete_report["total"] = {}
    for seed_idx, checkpoint_path in enumerate(analysis_config[args.model_type]['checkpoints']):
        temp_seed = args.base_seed + seed_idx

        print(f"Evaluating seed {seed_idx}...")
        set_seed(temp_seed)

        # Initialize model and load weights
        match args.model_type:
            case "repercent":
                disenEncoders = [make_model(model_config, data_config, modality=m + 1, M=data_config["create_data"]["M"]) for m in range(data_config["create_data"]["M"])]
                model = RePercENT(M=data_config["create_data"]["M"],
                                disenEncoder= disenEncoders,
                                disen_mapping= model_config["repercent"]["disen_mapping"]).to(device)
            case "gmlp":
                model = make_model_jointopt(model_config).to(device)
            case "gru":
                model = make_model_jointopt(model_config).to(device)
            case _:
                raise ValueError(f"Unsupported model type: {args.model_type}")
        
        temp_state_dict = torch.load(os.path.join(project_root, checkpoint_path), map_location=device)
        model.load_state_dict(temp_state_dict['model_state_dict'])



        model.to(device)
        
        # Evaluate on test set
        test_loader = DataLoader(test_dataset, batch_size= 32, shuffle=False, generator= torch.Generator().manual_seed(seed_idx))
        
        temp_metrics = evaluate_model(model, test_loader, device, M= M, comp_mod= args.comp_mod, component= args.component)
        
        # Store results for this seed
        complete_report[f"seed_{temp_seed}"] = temp_metrics

        # Aggregate results across seeds
        for split_name, split_metrics in temp_metrics.items():
            complete_report["total"].setdefault(split_name, {})
            for metric_name, metric_val in split_metrics.items():
                # convert numpy scalars / torch scalars to python float
                if hasattr(metric_val, "item"):
                    metric_val = metric_val.item()
                else:
                    metric_val = float(metric_val)

                complete_report["total"][split_name].setdefault(metric_name, [])
                complete_report["total"][split_name][metric_name].append(metric_val)
    
    # Evaluate mean and std values for each metric across seeds
    complete_report["summary"] = {}
    for split_name, metrics_dict in complete_report["total"].items():
        complete_report["summary"][split_name] = {}
        for metric_name, values in metrics_dict.items():
            values = np.array(values, dtype=np.float64)
            complete_report["summary"][split_name][metric_name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0
            }

    match args.comp_mod:
        case 1:
            name = f"{args.model_type}_evaluation_images_vs_captions_{args.component}"
        case 2:
            name = f"{args.model_type}_evaluation_images_vs_definitions_{args.component}"
        case 3:
            name = f"{args.model_type}_evaluation_images_vs_both_{args.component}"
    if args.component != "shared":
        name = f"{name}_{args.component}"
            
    wandb.init(
        project= f"irfl_{M}m_posthoc_analysis",
        name= name,
        config= analysis_config[args.model_type]
    )

    summary_table = wandb.Table(columns=["Split", "Metric", "Mean ± Std"])

    for split_name, metrics_dict in complete_report["summary"].items():
        for metric_name, stats in metrics_dict.items():
            mean = stats["mean"]
            std = stats["std"]

            summary_table.add_data(
                split_name,
                metric_name,
                f"{mean:.4f} ± {std:.4f}"
            )

    wandb.log({"Evaluation Summary": summary_table})
    wandb.finish()

if __name__ == "__main__":
    main()
