import torch
import numpy as np
from torch.utils.data import Dataset
# Adjust sys.path to import always from src
import os
import sys
from typing import Dict, Any, Callable, Optional, Sequence, Tuple, List, Union
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class MultimodalDataset(Dataset):
  def __init__(self, total_data, labels_1 = None, labels_2=None, labels_s=None):
    self.data = torch.from_numpy(total_data).float()
    self.num_modalities = self.data.shape[0]
    self.labels_1 = labels_1
    self.labels_2 = labels_2
    self.labels_s = labels_s
  
  def __len__(self):
    return self.data.shape[1]

  def __getitem__(self, idx):
    if self.labels_1 is not None:
        return tuple([self.data[i, idx] for i in range(self.num_modalities)] + [self.labels_1[idx]] + [self.labels_2[idx]] + [self.labels_s[idx]])
    else:
        return tuple([self.data[i, idx] for i in range(self.num_modalities)])
        
  def sample_batch(self, batch_size):
    sample_idxs = np.random.choice(self.__len__(), batch_size, replace=False)
    samples = self.__getitem__(sample_idxs)
    return samples


# This class handles the generation of synthetic dataset for the two-modality case
class GenerateData():
    def __init__(self, N_data: int, mod_type: str= "rbf", latent_dims: Dict[str, int]= {'Zs': 50, 'Z1': 50, 'Z2': 50}):
        """
        Args:
            N_data (int): Number of data samples to generate.
            mod_type (str): Type of modulation ('rbf', 'random'). This defines how each modality vectors with be projected across time.
            latent_dims (Dict[str, int]): Dictionary specifying the dimensions of shared and modality-specific latent factors.
        """
        self.N_data = N_data
        self.mod_type = mod_type
        self.latent_dims = latent_dims
        

    def rbf_mod(self, t: int, d: int, gamma: float = 10.0):
        time_points = np.linspace(0, 1, t)
        centers = np.linspace(0, 1, d)
        
        rbf_mat = np.exp(- gamma * (time_points[:, None] - centers[None, :])**2)
        return rbf_mat / rbf_mat.sum(axis=1, keepdims=True)

    def random_mod(self, t: int, d: int):
        random_mat = np.random.randn(t, d)
        return random_mat / np.linalg.norm(random_mat, axis=1, keepdims=True)

    def create_modulation_mats(self, t1: int = 5, t2: int = 5, gamma: float = 10.0):
        self.d1 = self.latent_dims['Z1'] + self.latent_dims['Zs'] # dimensionality for modality 1
        self.d2 = self.latent_dims['Z2'] + self.latent_dims['Zs'] # dimensionality for modality 2
        self.t1 = t1
        self.t2 = t2
        match self.mod_type:
            case "rbf":
                self.W1 = self.rbf_mod(t1, self.d1, gamma)
                self.W2 = self.rbf_mod(t2, self.d2, gamma)
            case "random":
                self.W1 = self.random_mod(t1, self.d1)
                self.W2 = self.random_mod(t2, self.d2)
            case _:
                raise ValueError("Unsupported modulation type")
        
        return self.W1, self.W2

    def generate_labels(self, latent_factor_t, seed: int = 0):
        """
        Converts a continuous latent factor (t_Z) into a binary categorical label (0/1)
        using a fixed, random linear projection and median thresholding.
        """
        torch.manual_seed(seed)
        # latent_factor_t is (N_samples, D_latent)
        d = latent_factor_t.shape[1]
        
        # Create a simple fixed linear projector (50D -> 1D)
        projector = nn.Linear(d, 1)
        
        # Freeze the weights (essential for fixed, ground-truth label)
        for param in projector.parameters():
            param.requires_grad = False
        
        # Apply projection and non-linearity (ReLU makes it non-trivially related)
        score_vector = projector(torch.relu(torch.Tensor(latent_factor_t))).numpy().flatten() 
        
        # Add small noise for robustness
        score_vector += np.random.normal(0, 0.01, score_vector.shape)
        
        # Thresholding based on the median (ensures a balanced 50/50 split of classes)
        midprob = np.median(score_vector)
        total_labels = (score_vector >= midprob).astype(int)
        
        return total_labels

    def create_dataset(self):
        self.set_seed(0)
        # Generate latent factors
        data = {}
        for k, d in self.latent_dims.items():
            # sample a random latent factor from a standard normal distribution
            data[k] = np.random.multivariate_normal(np.zeros((d,)), np.eye(d) * 0.5, size= self.N_data)
        
        t_Z1 = data['Z1']
        t_Zs = data['Zs']
        t_Z2 = data['Z2']

        if not hasattr(self, 'W1') or not hasattr(self, 'W2'):
            print("Modulation matrices not found, creating with default parameters for each modality.")
            self.create_modulation_mats()

        # generate modulation matrices
        Z1 = np.concatenate((t_Z1, t_Zs), axis=-1)  # Latent representation for modality 1
        Z2 = np.concatenate((t_Z2, t_Zs), axis=-1)  # Latent representation for modality 2


        X1 = Z1[:, None, :] * self.W1[None, :, :] # Modulated data for modality 1
        X2 = Z2[:, None, :] * self.W2[None, :, :] # Modulated data for modality 2
        
        # --- D. Generate Disentanglement Labels (Y1, Y2, Ys) ---
    
        # Y1: Derived ONLY from t_Z1 (Specific 1)
        labels_1 = self.generate_labels(t_Z1, seed= np.random.randint(0, 10000))
        
        # Ys: Derived ONLY from t_Zs (Shared)
        labels_s = self.generate_labels(t_Zs, seed= np.random.randint(0, 10000))
        
        # Y2: Derived ONLY from t_Z2 (Specific 2)
        labels_2 = self.generate_labels(t_Z2, seed= np.random.randint(0, 10000))
        
        total_data = np.array([X1, X2])  # Shape: (2, N_samples, t, d)
        

        # pack into a dictionary
        self.dataset_dict = {
            'total_data': total_data,
            'labels_1': labels_1,
            'labels_2': labels_2,
            'labels_s': labels_s,
            't_Z1': t_Z1,
            't_Zs': t_Zs,
            't_Z2': t_Z2
        }
        return self.dataset_dict

    def print_dataset_info(self):
        if not hasattr(self, 'dataset_dict'):
            raise ValueError("Dataset not created yet. Please run create_dataset() first.")
        
        print("Dataset Information:")
        print(f"Number of samples: {self.N_data}")
        print(f"Modality 1 data shape: {self.dataset_dict['total_data'][0].shape}")
        print(f"Modality 2 data shape: {self.dataset_dict['total_data'][1].shape}")
        print(f"Labels 1 shape: {self.dataset_dict['labels_1'].shape}, Unique classes: {np.unique(self.dataset_dict['labels_1'])}")
        print(f"Labels 2 shape: {self.dataset_dict['labels_2'].shape}, Unique classes: {np.unique(self.dataset_dict['labels_2'])}")
        print(f"Labels s shape: {self.dataset_dict['labels_s'].shape}, Unique classes: {np.unique(self.dataset_dict['labels_s'])}")

    # configure seed for all randomness
    @staticmethod   
    def set_seed(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)



    
