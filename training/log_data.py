import wandb
import os

# This module provides helper functions for logging model details, checkpoints, 
# and datasets to Weights & Biases (WandB) during training and evaluation. 


def log_model_details(run, model_name, data_config, model_config, training_config):
    """
    Logs the model architecture, training configuration, and data used to WandB.
    Args:
        model_name: Name of the model to be logged.
        data_config: Configuration dictionary for the data.
        model_config: Configuration dictionary for the model.
        training_config: Configuration dictionary for the training process.
    """
    details = wandb.Artifact(
        name= f"{run.name}_details",
        type= "run_details"
    )
    details.add_file(data_config)
    details.add_file(model_config)
    details.add_file(training_config)
    run.log_artifact(details)




def log_model_checkpoint(run, checkpoint_path, epoch, extra_meta= None):
    """
    Logs the model checkpoint to WandB.
    Args:
        run: WandB run object.
        checkpoint_path: Path to the model checkpoint file.
        epoch: Current epoch number.
    """
    checkpoint_artifact = wandb.Artifact(
        name= f"{run.name}_checkpoint_epoch_{epoch}",
        type= "model_checkpoint",
        metadata={"epoch": epoch, **(extra_meta or {}), "best_overall": extra_meta.get("best_overall", False) if extra_meta else False}
    )
    checkpoint_artifact.add_file(checkpoint_path)
    run.log_artifact(checkpoint_artifact)




def log_dataset(dataset_name, dataset_path, data_config_path):
    """
    Logs the dataset used for training to WandB.
    Args:
        dataset_name: Name of the dataset.
        dataset_path: Path to the dataset file.
        data_config_path: Path to the data configuration file.
    """
    dataset_artifact = wandb.Artifact(
        name= dataset_name,
        type= "dataset"
    )
    dataset_artifact.add_file(os.path.join(dataset_path, f"{dataset_name}", "dataset.pt"))
    dataset_artifact.add_file(os.path.join(dataset_path, f"{dataset_name}", "README.md"))
    dataset_artifact.add_file(data_config_path)
    wandb.log_artifact(dataset_artifact)