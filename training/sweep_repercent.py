import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import yaml
import wandb
from src.utils.synthetic_dataset import GenerateData, MultimodalDataset
from src.models.repercent import DisenLoss, RePercENT
from training.train_repercent import make_dataloaders, train, make_model


def train_sweep():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load base configs
    with open(os.path.join(script_dir, "..", "configs", "data", "synthetic_data.yaml")) as f:
        data_config = yaml.safe_load(f)
    with open(os.path.join(script_dir, "..", "configs", "model", "repercent.yaml")) as f:
        model_config = yaml.safe_load(f)
    with open(os.path.join(script_dir, "..", "configs", "training", "train_synthetic.yaml")) as f:
        training_config = yaml.safe_load(f)

    run = wandb.init(project="repercent_synthetic_sweep", name= "sweep_dataset10")
    cfg = wandb.config

    # Apply sweep overrides
    training_config["optimizer"]["lr"] = cfg.learning_rate
    training_config["training"]["n_epochs"] = cfg.num_epochs
    training_config["disen_loss"]["lmd"] = cfg.lmd
    training_config["disen_loss"]["alpha"] = cfg.alpha
    model_config["perceiver"]["cross_heads"] = cfg.cross_heads
    model_config["perceiver"]["latent_heads"] = cfg.latent_heads
    model_config["perceiver"]["depth"] = cfg.depth

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset_dir = os.path.join(script_dir, 'data', 'repercent_synthetic', 'dataset10')

    # Load the train and test datasets
    data_split = torch.load(os.path.join(dataset_dir, 'data_split.pt'), weights_only=False)
    train_dataset = data_split['train_dataset']
    # split train to train and val
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


    train_loader, val_loader = make_dataloaders(train_dataset, val_dataset, training_config["training"])

    # Model + loss + optimizer
    disen_m1 = make_model(model_config, data_config, modality="m1")
    disen_m2 = make_model(model_config, data_config, modality="m2")
    model = RePercENT(M=2, disenEncoder=[disen_m1, disen_m2]).to(device)

    disen_loss = DisenLoss(
        alpha=training_config["disen_loss"]["alpha"],
        lmd=training_config["disen_loss"]["lmd"],
        lmd_end_value=training_config["disen_loss"]["lmd_end_value"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["optimizer"]["lr"],
        weight_decay=training_config["optimizer"]["weight_decay"],
    )

    checkpoint_dir = os.path.join(script_dir, "..", "checkpoints", "repercent_synthetic", run.name)
    train(
        gen_data,
        train_loader,
        test_loader,
        model,
        optimizer,
        disen_loss,
        training_config["training"]["n_epochs"],
        device,
        checkpoint_dir=checkpoint_dir,
    )
    wandb.finish()


def sweep_repercent():    # Define the sweep configuration
    


    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project='repercent_synthetic_sweep')

    # Define the training function to be called by wandb agent
    #TODO: modify train_sweep to accept hyperparameters from wandb.config

    # Start the sweep agent
    wandb.agent(sweep_id, function=train_sweep, count=18)

