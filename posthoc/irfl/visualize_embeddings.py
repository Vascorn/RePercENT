import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from posthoc.plotting_config import apply_paper_plot_style
from src.models import repercent, jointopt
from src.models.repercent import RePercENT
from src.utils.irfl_dataset import make_dataset
from training.train_repercent import make_dataloaders, make_model
from training.train_jointopt import make_model_jointopt
import wandb
from src.utils.helpers import load_yaml, set_seed
from torch.utils.data import DataLoader
from posthoc.irfl.helper_vis import plot_embeddings, extract_all_embeddings

apply_paper_plot_style()



def main():
    parser = argparse.ArgumentParser(description="Visualize embeddings of unique and shared components from the three different modalities (images, captions, definitions) for the IRFL dataset using dimensionality reduction techniques like PCA, t-SNE, or UMAP.")
    parser.add_argument('--datasets_path', type=str, default="../../data/irfl/datasets/", help='Path to the directory containing the IRFL dataset tensors wrt to this script')
    parser.add_argument('--model_type', type=str, choices=['repercent', 'gmlp'], default='repercent', help='Type of model to train, for now only repercent is implemented')
    parser.add_argument('--comp_mod', type=int, choices=[1, 2], default= 1, help='Which modality to compute similarities for (1 for captions, 2 for definitions, 3 for adding \
                                                                                    the similarities between images- captions and images - definitions and then comparing the metrics). \
                                                                                    Note that 2 and 3 is only relevant for the 3-modality setting')
    # Define number of splits and seeds
    parser.add_argument('--select_seed', type=int, default= 1, help='Select the seed index to visualize (0-based index, should be less than n_seeds)')
    args = parser.parse_args()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    M = 3 # number of modalities, for the IRFL: M = 2 -> images + captions, M = 3 -> images + captions + definitions
    
    # Loading configurations for data, model, and training
    print("Loading configurations...")
    data_config_path = os.path.join(script_dir, "../..", "configs", "data", f"irfl_data_{M}m.yaml")
    data_config = load_yaml(data_config_path)

    model_config_path = os.path.join(script_dir, "../..", "configs", "model", f"{args.model_type}_irfl_{M}m.yaml")
    model_config = load_yaml(model_config_path)

    analysis_config_path = os.path.join(script_dir, "../..", "configs", "posthoc_analysis", f"irfl_{M}m.yaml")
    analysis_config = load_yaml(analysis_config_path)
    
    # Check that the requested seed index is valid
    n_seeds = analysis_config['hyperparameters']['n_seeds']
    assert 0 <= args.select_seed < n_seeds, f"select_seed should be between 0 and {n_seeds - 1}, but got {args.select_seed}"

    # Load the *full dataset once*
    print("Loading datasets...")
    
    total_test_data = torch.load(os.path.join(script_dir, args.datasets_path, 'IRFL_test_tensors_2.pt'), map_location="cpu")

    total_test_data_aug = torch.load(os.path.join(script_dir, args.datasets_path, 'IRFL_test_tensors_aug_2.pt'), map_location="cpu")
    test_dataset, test_data_dict = make_dataset(total_data= total_test_data | total_test_data_aug, num_modalities= data_config["create_data"]["M"], data_type='test', include_original=True)

    print(f"Analysis config: {analysis_config[args.model_type]['checkpoints']}")

    # define project root for loading checkpoints
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    checkpoint_path = analysis_config[args.model_type]['checkpoints'][args.select_seed]
    print(f"Loading model from checkpoint: {checkpoint_path}")
    # init device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init results storage
  
    set_seed(2)

    # Initialize model and load weights
    match args.model_type:
        case "repercent":
            disenEncoders = [make_model(model_config, data_config, modality=m + 1, M=data_config["create_data"]["M"]) for m in range(data_config["create_data"]["M"])]
            model = RePercENT(M=data_config["create_data"]["M"],
                            disenEncoder= disenEncoders,
                            disen_mapping= model_config["repercent"]["disen_mapping"],
                            vmfkappa=model_config["repercent"]["vmfkappa"]).to(device)
        case "gmlp":
            model = make_model_jointopt(model_config).to(device)
        case _:
            raise ValueError(f"Unsupported model type: {args.model_type}")
    
    temp_state_dict = torch.load(os.path.join(project_root, checkpoint_path), map_location=device)
    model.load_state_dict(temp_state_dict['model_state_dict'])



    model.to(device)
    
    
    test_loader = DataLoader(test_dataset, batch_size= 32, shuffle=False, generator= torch.Generator().manual_seed(2))
    fig_dir = os.path.join(script_dir, "figures/embeddings/")
    os.makedirs(fig_dir, exist_ok=True)
    embeddings_all = extract_all_embeddings(model, test_loader, device, M= M, comp_mod= args.comp_mod)
    plot_embeddings(embeddings_all, method="umap", f_type= "all", random_state= args.select_seed, dim= 2, fig_path= os.path.join(fig_dir, f"embeddings_{args.model_type}_seed{args.select_seed}.pdf"))


if __name__ == "__main__":
    main()


