import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import yaml
import wandb
from src.utils.synthetic_dataset import GenerateData, MultimodalDataset
from src.models.repercent import DisenLoss, RePercENT
from training.train_repercent import make_dataloaders, train, make_model, train_loop, test_loop
from src.utils.helpers import linear_probe, extract_latents_and_labels
import numpy as np 


def train_sweep(config= None):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load base configs
    with open(os.path.join(script_dir, "..", "configs", "data", "synthetic_data.yaml")) as f:
        data_config = yaml.safe_load(f)
    with open(os.path.join(script_dir, "..", "configs", "model", "repercent.yaml")) as f:
        model_config = yaml.safe_load(f)
    with open(os.path.join(script_dir, "..", "configs", "training", "train_synthetic.yaml")) as f:
        training_config = yaml.safe_load(f)


    # Initialize the dataset
    dataset_dir = os.path.join(script_dir, '..', 'data', 'repercent_synthetic', 'dataset11')

    # Load the train and test datasets
    data_split = torch.load(os.path.join(dataset_dir, 'sweep_data_split.pt'), weights_only=False)
    train_dataset = data_split['train_dataset']
    val_dataset = data_split['val_dataset']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with wandb.init(config= config):
        cfg = wandb.config

        # Create dataloaders
        train_loader = Dataloader(train_dataset, batch_size= cfg.batch_size, shuffle=True)
        val_loader = Dataloader(val_dataset, batch_size= cfg.batch_size, shuffle=False)

        # Override model config with sweep parameters
        model_config['transformer']['cross_heads'] = cfg.cross_heads
        model_config['transformer']['latent_heads'] = cfg.latent_heads
        model_config['transformer']['depth'] = cfg.depth

        # Model + loss + optimizer
        disen_m1 = make_model(model_config, data_config, modality="m1")
        disen_m2 = make_model(model_config, data_config, modality="m2")
        model = RePercENT(M=2, disenEncoder=[disen_m1, disen_m2]).to(device)

        disen_loss = DisenLoss(
            alpha= cfg.alpha,
            lmd= cfg.lmd,
            lmd_end_value= 0.0,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr= cfg.learning_rate,
            weight_decay= training_config["optimizer"]["weight_decay"],
        )
        for epoch in range(cfg.num_epochs):
            epoch_loss, epoch_loss_val = 0.0, 0.0
            epoch_ortho_loss, epoch_ortho_loss_val = 0.0, 0.0
            epoch_unique_loss, epoch_unique_loss_val = 0.0, 0.0
            epoch_shared_loss, epoch_shared_loss_val = 0.0, 0.0

            model.train()
            for batch_idx, (data_m1, data_m2, _, _, _) in enumerate(train_loader):
                temp_b = data_m1.shape[0]
                data_m1 = data_m1.to(device)
                data_m2 = data_m2.to(device)
            
                # Augment data 
                data_m1_aug = GenerateData.augment_data(data_m1, aug_type="random")
                data_m2_aug = GenerateData.augment_data(data_m2, aug_type="random")
                
                # Forward pass through RePercENT
                loss_train, logs_train = train_loop(data_m1, data_m2, data_m1_aug, data_m2_aug, model, optimizer, disen_loss)
                # Accumulate losses
                epoch_loss += loss_train.item() / temp_b
                epoch_ortho_loss += logs_train['ortho'] / temp_b
                epoch_unique_loss += logs_train['unique'] / temp_b
                epoch_shared_loss += logs_train['shared'] / temp_b
                
            # Validation loop
            model.eval()
            with torch.no_grad():
                for batch_idx, (data_m1, data_m2, _, _, _) in enumerate(val_loader):
                    temp_b = data_m1.shape[0]
                    data_m1 = data_m1.to(device)
                    data_m2 = data_m2.to(device)
                    
                    # Augment data 
                    data_m1_aug = GenerateData.augment_data(data_m1, aug_type="random")
                    data_m2_aug = GenerateData.augment_data(data_m2, aug_type="random")
                    
                    # Forward pass through RePercENT
                    loss_test, logs_test = test_loop(data_m1, data_m2, data_m1_aug, data_m2_aug, model, disen_loss)

                    # Accumulate losses
                    epoch_loss_val += loss_test.item() / temp_b
                    epoch_ortho_loss_val += logs_test['ortho'] / temp_b
                    epoch_unique_loss_val += logs_test['unique'] / temp_b
                    epoch_shared_loss_val += logs_test['shared'] / temp_b
            # Log metrics and linear probe accuracy to W&B
            train_data_dict = extract_latents_and_labels(model, train_loader, device)
            val_data_dict = extract_latents_and_labels(model, val_loader, device)
            wandb.log({
                "Train Loss - total": epoch_loss / len(train_loader),
                "Train Loss - ortho": epoch_ortho_loss / len(train_loader),
                "Train Loss - unique": epoch_unique_loss / len(train_loader),
                "Train Loss - shared": epoch_shared_loss / len(train_loader),
                "Val Loss - total": epoch_loss_val / len(val_loader),
                "Val Loss - ortho": epoch_ortho_loss_val / len(val_loader),
                "Val Loss - unique": epoch_unique_loss_val / len(val_loader),
                "Val Loss - shared": epoch_shared_loss_val / len(val_loader),
                "epoch": epoch + 1,
                "Linear Probe Acc - u_12 -> labels_1": linear_probe(train_data_dict['u_12'], train_data_dict['labels_1'], val_data_dict['u_12'], val_data_dict['labels_1']),
                "Linear Probe Acc - u_12 -> labels_s": linear_probe(train_data_dict['u_12'], train_data_dict['labels_s'], val_data_dict['u_12'], val_data_dict['labels_s']),
                "Linear Probe Acc - u_21 -> labels_2": linear_probe(train_data_dict['u_21'], train_data_dict['labels_2'], val_data_dict['u_21'], val_data_dict['labels_2']),
                "Linear Probe Acc - u_21 -> labels_s": linear_probe(train_data_dict['u_21'], train_data_dict['labels_s'], val_data_dict['u_21'], val_data_dict['labels_s']),
                "Linear Probe Acc - s -> labels_1": linear_probe(np.concatenate((train_data_dict['s_21'], train_data_dict['s_12']), axis= -1), train_data_dict['labels_1'], np.concatenate((val_data_dict['s_21'], val_data_dict['s_12']), axis= -1), val_data_dict['labels_1']),
                "Linear Probe Acc - s -> labels_2": linear_probe(np.concatenate((train_data_dict['s_21'], train_data_dict['s_12']), axis= -1), train_data_dict['labels_2'], np.concatenate((val_data_dict['s_21'], val_data_dict['s_12']), axis= -1), val_data_dict['labels_2']),
                "Linear Probe Acc - s -> labels_s": linear_probe(np.concatenate((train_data_dict['s_21'], train_data_dict['s_12']), axis= -1), train_data_dict['labels_s'], np.concatenate((val_data_dict['s_21'], val_data_dict['s_12']), axis= -1), val_data_dict['labels_s']),
            })



def main():   
    # Create and save the sweep dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read the configuration files for data
    data_config_path = os.path.join(script_dir, "..", "configs", "data", "synthetic_data.yaml")
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
        
         
    # Define the sweep configuration
    with open(os.path.join(os.path.dirname(__file__), "..", "configs", "sweeps", "synthetic_sweep.yaml")) as f:
        sweep_config = yaml.safe_load(f)
    print(f"Sweep Configuration: {sweep_config}")
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project='repercent_synthetic_sweep')

    # Start the sweep agent
    wandb.agent(sweep_id, function=train_sweep, count= 20)

if __name__ == "__main__":
    main()