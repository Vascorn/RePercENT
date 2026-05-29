"""Demo training script for RePercENT on randomly generated multimodal data."""

import argparse
import copy
import os
import random
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from src.models.repercent import RePercENT, DisenLoss
from src.utils.helpers import load_yaml, set_seed
from training.train_repercent import make_dataloaders, make_model


class MockMultimodalDataset(Dataset):
    def __init__(self, X_by_modality: dict[int, torch.Tensor], aug_noise_std: float = 0.05):
        """
        A mock dataset where each modality has shape (N, sequence_length, feature_dim).

        Args:
            X_by_modality: A dictionary mapping modality indices to their corresponding data tensors. Each tensor should have shape (N, ts, dm).
            aug_noise_std: Standard deviation of the Gaussian noise added for data augmentation.
        """
        self.X_by_modality = X_by_modality
        self.aug_noise_std = aug_noise_std
        self.modalities = sorted(X_by_modality.keys())
        self.n_samples = X_by_modality[self.modalities[0]].shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        X = {m: self.X_by_modality[m][idx].clone() for m in self.modalities}
        X_aug = {
            m: X[m] + torch.randn_like(X[m]) * self.aug_noise_std
            for m in self.modalities
        }
        return X, X_aug


def create_mock_data(data_config: dict = None, seed: int = 0) -> Dataset:
    """
    Create a mock dataset with random data for testing purposes.

    Args:
        data_config: Configuration dictionary for the data.
        seed: Random seed for reproducibility.
    """
    if data_config is None or "create_data" not in data_config:
        raise ValueError("`data_config` must contain a `create_data` section.")

    print("Creating mock dataset...")
    cfg = data_config["create_data"]
    num_modalities = cfg["M"]
    n_data = cfg["N_data"]
    latent_dim = cfg["latent_dim"]
    sequence_lengths = cfg["ts"]
    normalize = cfg.get("normalize", False)
    aug_noise_std = cfg.get("sigma", 0.05)

    if len(sequence_lengths) != num_modalities:
        raise ValueError(
            f"Expected `ts` to have length {num_modalities}, got {len(sequence_lengths)}."
        )

    # Match the feature dimensionality used by the synthetic RePercENT setup.
    feature_dim = (2 ** (num_modalities - 1)) * latent_dim

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    X_by_modality = {}
    for modality_idx, sequence_dim in enumerate(sequence_lengths):
        modality_data = torch.randn(n_data, sequence_dim, feature_dim, dtype=torch.float32)

        if normalize:
            mean = modality_data.mean(dim=(0, 1), keepdim=True)
            std = modality_data.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
            modality_data = (modality_data - mean) / std

        X_by_modality[modality_idx] = modality_data

    dataset = MockMultimodalDataset(X_by_modality=X_by_modality, aug_noise_std=aug_noise_std)

    print(
        f"Created mock dataset with {len(dataset)} samples, {num_modalities} modalities, "
        f"feature dimension {feature_dim}, and sequence dimensions {sequence_lengths}."
    )
    return dataset


def split_dataset_seeded(dataset, test_size: float, val_size: float, seed: int):
    n_total = len(dataset)
    n_test = int(round(n_total * test_size))
    n_val = int(round(n_total * val_size))
    n_train = n_total - n_test - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_test, n_val], generator=g)


def _empty_masks(modalities: list[torch.Tensor]) -> list[None]:
    return [None for _ in modalities]


def train_loop(X, X_aug, model, optimizer, disen_loss):
    """
    Single Epoch training step for RePercENT model
    Args:
        X: Batch data from all modalities
        X_aug: Augmented batch data from all modalities
        model: RePercENT model in training mode
        optimizer: Optimizer for RePercENT model
        disen_loss: Disentanglement loss function
    Returns:
        loss: Computed loss value for the batch
        logs: Dictionary containing loss components for monitoring
    """
    outputs = model(X, mask=_empty_masks(X))
    outputs_aug = model(X_aug, mask=_empty_masks(X_aug))

    loss, logs = disen_loss(outputs, outputs_aug)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, logs


def test_loop(X, X_aug, model, disen_loss):
    """
    Single Epoch testing step for RePercENT model
    Args:
        X: Batch data from all modalities
        X_aug: Augmented batch data from all modalities
        model: RePercENT model in evaluation mode
        disen_loss: Disentanglement loss function
    Returns:
        loss: Computed loss value for the batch
        logs: Dictionary containing loss components for monitoring
    """
    outputs = model(X, mask=_empty_masks(X))
    outputs_aug = model(X_aug, mask=_empty_masks(X_aug))

    loss, logs = disen_loss(outputs, outputs_aug)

    return loss, logs


def train_mock(
    train_loader,
    test_loader,
    model,
    optimizer,
    disen_loss,
    epochs,
    device,
    val_loader=None,
    checkpoint_dir="./checkpoints",
):
    """
    Train RePercENT on the mock dataset with optional validation and checkpointing.

    Args:
        train_loader: DataLoader for training dataset
        test_loader: DataLoader for test dataset.
        model: RePercENT model
        optimizer: Optimizer for RePercENT model
        disen_loss: Disentanglement loss function
        epochs: Number of training epochs
        device: Device to run the training on (CPU/GPU)
        val_loader: Optional DataLoader for validation.
        checkpoint_dir: Directory to save model checkpoints

    Returns:
        final_metrics: Final test metrics when validation checkpoints are used, otherwise None.
    """
    torch.cuda.empty_cache()
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters: {num_parameters}")
    num_modalities = disen_loss.M

    best_fw_val_loss = float('inf')
    best_model_state = None
    final_metrics = None
    print("\nStarting mock training...")
    for _iter in range(epochs):
        epoch_loss = 0.0
        epoch_ortho_loss = 0.0
        epoch_unique_loss = 0.0
        epoch_shared_loss = 0.0
        epoch_fw_loss = 0.0
        model.train()

        print(f"----- Epoch: {_iter + 1} / {epochs} -----")

        for X, X_aug in train_loader:
            X = [X[m].to(device) for m in range(num_modalities)]
            X_aug = [X_aug[m].to(device) for m in range(num_modalities)]

            loss, loss_logs = train_loop(X, X_aug, model, optimizer, disen_loss)

            epoch_loss += loss.item()
            epoch_ortho_loss += loss_logs['ortho']
            epoch_unique_loss += loss_logs['unique']
            epoch_shared_loss += loss_logs['shared']
            epoch_fw_loss += loss_logs['fw_loss']

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_ortho_loss = epoch_ortho_loss / len(train_loader)
        avg_unique_loss = epoch_unique_loss / len(train_loader)
        avg_shared_loss = epoch_shared_loss / len(train_loader)
        avg_fw_loss = epoch_fw_loss / len(train_loader)

        print(
            f"Training  Loss: {avg_epoch_loss:.5f} | Ortho: {avg_ortho_loss:.5f} "
            f"| Unique: {avg_unique_loss:.5f} | Shared: {avg_shared_loss:.5f} "
            f"| Lmd: {disen_loss.lmd:.6f}, alpha: {disen_loss.alpha:.6f}"
        )

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_epoch_loss = 0.0
                val_epoch_ortho_loss = 0.0
                val_epoch_unique_loss = 0.0
                val_epoch_shared_loss = 0.0
                val_epoch_fw_loss = 0.0
                for X, X_aug in val_loader:
                    X = [X[m].to(device) for m in range(num_modalities)]
                    X_aug = [X_aug[m].to(device) for m in range(num_modalities)]

                    loss_val, logs_val = test_loop(X, X_aug, model, disen_loss)

                    val_epoch_loss += loss_val.item()
                    val_epoch_ortho_loss += logs_val["ortho"]
                    val_epoch_unique_loss += logs_val["unique"]
                    val_epoch_shared_loss += logs_val["shared"]
                    val_epoch_fw_loss += logs_val["fw_loss"]

            avg_epoch_loss_val = val_epoch_loss / len(val_loader)
            avg_ortho_loss_val = val_epoch_ortho_loss / len(val_loader)
            avg_unique_loss_val = val_epoch_unique_loss / len(val_loader)
            avg_shared_loss_val = val_epoch_shared_loss / len(val_loader)
            avg_fw_loss_val = val_epoch_fw_loss / len(val_loader)

            print(
                f"Validation  Loss: {avg_epoch_loss_val:.5f} | Ortho: {avg_ortho_loss_val:.5f} "
                f"| Unique: {avg_unique_loss_val:.5f} | Shared: {avg_shared_loss_val:.5f} "
                f"| Fixed Weight Loss: {avg_fw_loss_val:.5f}"
            )

            if avg_fw_loss_val < best_fw_val_loss:
                print(
                    f"New best model found at epoch {_iter + 1} with validation fixed weight loss: "
                    f"{avg_fw_loss_val:.5f} (previous best: {best_fw_val_loss:.5f})"
                )
                best_fw_val_loss = avg_fw_loss_val
                best_model_state = copy.deepcopy(model.state_dict())

        if (_iter + 1) % 10 == 0 or (_iter + 1) == epochs:
            checkpoint_name = (
                f"checkpoint_epoch_{_iter + 1}.pt"
                if (_iter + 1) // 10 != (epochs // 10)
                else "final_checkpoint.pt"
            )
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                'epoch': _iter + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }

            torch.save(checkpoint, checkpoint_path)

            print(f"Model checkpoint saved at {checkpoint_path}")

        if (_iter + 1) == epochs and best_model_state is not None:
            checkpoint_name = "final_best_checkpoint.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                'epoch': _iter + 1,
                'model_state_dict': best_model_state
            }

            torch.save(checkpoint, checkpoint_path)

            print(f"Best model checkpoint saved at {checkpoint_path}")
            if test_loader is not None:
                model.eval()

                with torch.no_grad():
                    test_epoch_loss = 0.0
                    test_epoch_ortho_loss = 0.0
                    test_epoch_unique_loss = 0.0
                    test_epoch_shared_loss = 0.0
                    test_epoch_fw_loss = 0.0
                    for X, X_aug in test_loader:
                        X = [X[m].to(device) for m in range(num_modalities)]
                        X_aug = [X_aug[m].to(device) for m in range(num_modalities)]

                        loss_val, logs_val = test_loop(X, X_aug, model, disen_loss)

                        test_epoch_loss += loss_val.item()
                        test_epoch_ortho_loss += logs_val["ortho"]
                        test_epoch_unique_loss += logs_val["unique"]
                        test_epoch_shared_loss += logs_val["shared"]
                        test_epoch_fw_loss += logs_val["fw_loss"]

                avg_epoch_loss_test = test_epoch_loss / len(test_loader)
                avg_ortho_loss_test = test_epoch_ortho_loss / len(test_loader)
                avg_unique_loss_test = test_epoch_unique_loss / len(test_loader)
                avg_shared_loss_test = test_epoch_shared_loss / len(test_loader)
                avg_fw_loss_test = test_epoch_fw_loss / len(test_loader)

                final_metrics = {
                    "avg_epoch_loss_test": avg_epoch_loss_test,
                    "avg_ortho_loss_test": avg_ortho_loss_test,
                    "avg_unique_loss_test": avg_unique_loss_test,
                    "avg_shared_loss_test": avg_shared_loss_test,
                    "avg_fw_loss_test": avg_fw_loss_test,
                }
                print(
                    f"Test  Loss: {avg_epoch_loss_test:.5f} | Ortho: {avg_ortho_loss_test:.5f} "
                    f"| Unique: {avg_unique_loss_test:.5f} | Shared: {avg_shared_loss_test:.5f} "
                    f"| Fixed Weight Loss: {avg_fw_loss_test:.5f}"
                )

    print("Mock training completed!\n")
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Train RePercENT on mock multimodal data")
    parser.add_argument('--base_seed', type=int, default=2, help='Base seed for reproducibility')
    parser.add_argument('--M', type=int, default=2, help='Number of modalities')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ###################### Load configurations #################################################
    data_config_path = os.path.join(
        script_dir, '..', 'configs', 'data', f'demo_data_{args.M}m.yaml'
    )
    model_config_path = os.path.join(
        script_dir, '..', 'configs', 'model', f'repercent_demo_{args.M}m.yaml'
    )
    training_config_path = os.path.join(
        script_dir, '..', 'configs', 'training', f'train_demo_{args.M}m.yaml'
    )

    data_config = load_yaml(data_config_path)
    model_config = load_yaml(model_config_path)
    training_config = load_yaml(training_config_path)

    num_modalities = data_config["create_data"]["M"]

    ###################### Create mock dataset and dataloaders ###############################
    dataset = create_mock_data(data_config=data_config, seed=args.base_seed)

    train_dataset, test_dataset, val_dataset = split_dataset_seeded(
        dataset,
        test_size=training_config["training"]["test_size"],
        val_size=training_config["training"]["val_size"],
        seed=args.base_seed,
    )

    train_seed = args.base_seed
    set_seed(train_seed)
    generator = torch.Generator().manual_seed(train_seed)

    train_id = time.strftime("%Y.%m.%d-%H:%M:%S") + f"_M_{args.M}_seed_{args.base_seed}"
    train_loader, test_loader, val_loader = make_dataloaders(
        train_dataset,
        test_dataset,
        val_dataset=val_dataset,
        batch_size=training_config["training"]["batch_size"],
        generator=generator,
    )

    ###################### Initialize RePercENT model and loss ###############################
    disen_encoders = [
        make_model(model_config, data_config, modality=m + 1, M=num_modalities)
        for m in range(num_modalities)
    ]
    model = RePercENT(
        M=num_modalities,
        disenEncoder=disen_encoders,
        disen_mapping=model_config["repercent"]["disen_mapping"],
        vmfkappa=model_config["repercent"]["vmfkappa"],
    ).to(device)

    disen_loss = DisenLoss(
        alpha=training_config["disen_loss"]["alpha"],
        beta=training_config["disen_loss"]["beta"],
        lmd=training_config["disen_loss"]["lmd"],
        lmd_start_value=training_config["disen_loss"]["lmd_start_value"],
        lmd_end_value=training_config["disen_loss"]["lmd_end_value"],
        lmd_n_iterations=training_config["disen_loss"]["lmd_n_iterations"],
        lmd_start_iteration=training_config["disen_loss"]["lmd_start_iteration"],
        ortho_norm=training_config["disen_loss"]["ortho_norm"],
        M=num_modalities,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["optimizer"]["lr"],
        weight_decay=training_config["optimizer"]["weight_decay"],
    )

    ###################### Train loop ######################################################
    checkpoint_dir = os.path.join(script_dir, '..', 'checkpoints', 'mock_data', train_id)
    final_metrics = train_mock(
        train_loader,
        test_loader,
        model,
        optimizer,
        disen_loss,
        training_config["training"]["n_epochs"],
        device,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
    )

    print("Final Metrics:", final_metrics)


if __name__ == "__main__":
    main()
