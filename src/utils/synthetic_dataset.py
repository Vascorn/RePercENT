import torch
import numpy as np
from torch.utils.data import Dataset
# Adjust sys.path to import always from src
import os
import sys
import torch
from typing import Dict, Any, Callable, Optional, Sequence, Tuple, List, Union, Literal
import torch.nn as nn
from scipy.stats import vonmises_fisher, multivariate_normal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def save_dataset(dataset, save_path: str, data_config: Dict[str, Any]= None):
     # create directory if it doesn't exist
    
    os.makedirs(os.path.dirname(save_path + "/dataset.pt"), exist_ok=True)
    torch.save(dataset, os.path.join(save_path, "dataset.pt"))
    print(f"Dataset saved at {save_path}")
    # create a README file with the data configuration
    if data_config is not None:
        readme_path = os.path.join(save_path, "README.md")
        with open(readme_path, 'w') as f:
                f.write("### Dataset Configuration\n\n")
                for key, value in data_config["create_data"].items():
                    if isinstance(value, dict):
                        print(f"Writing config section: {key}")
                        f.write(f"* {key}: \n\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"  - **{sub_key}**: {sub_value}\n")
                        f.write("\n")
                    else:
                        f.write(f"* **{key}**: {value}\n")
        print(f"README saved at {readme_path}")
    return

def save_data_split(train_dataset, test_dataset, save_path: str):
    os.makedirs(save_path, exist_ok=True)
    torch.save({'train_dataset': train_dataset, 'test_dataset': test_dataset}, os.path.join(save_path, "data_split.pt"))
    print(f"Data split saved at {save_path}")
    return

class MultimodalDataset(Dataset):
  def __init__(self, total_data, labels_1 = None, labels_2=None, labels_s=None):
    self.data = total_data
    self.num_modalities = len(self.data[0])
    self.labels_1 = labels_1
    self.labels_2 = labels_2
    self.labels_s = labels_s
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if self.labels_1 is not None:
        return tuple([torch.from_numpy(self.data[idx][i]).to(torch.float32) for i in range(self.num_modalities)] + [self.labels_1[idx]] + [self.labels_2[idx]] + [self.labels_s[idx]])
    else:
        return tuple([torch.from_numpy(self.data[idx][i]).to(torch.float32) for i in range(self.num_modalities)])
        
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
            mod_type (str): Type of modulation ('rbf', 'random', 'identity'). This defines how each modality vectors with be projected across time.
            latent_dims (Dict[str, int]): Dictionary specifying the dimensions of shared and modality-specific latent factors.
        """
        self.N_data = N_data
        self.mod_type = mod_type
        self.latent_dims = latent_dims
        

    def rbf_mod(self, t: int, d: int, gamma: float = 10.0, seed: int = 0):
        
        time_points = np.linspace(0, 1, t)
        centers = np.linspace(0, 1, d)
        
        rbf_mat = np.exp(- gamma * (time_points[:, None] - centers[None, :])**2)
        return rbf_mat / rbf_mat.sum(axis=1, keepdims=True)

    def random_mod(self, t: int, d: int):
        random_mat = torch.normal(mean=0.0, std=1.0, size=(t, d)).numpy()
        return random_mat 

    def create_modulation_mats(self, t1: int = 5, t2: int = 5, gamma1: float = 10.0, gamma2: float = 10.0):
        self.d1 = self.latent_dims['Z1'] # modality 1 specific latent dim
        self.d2 = self.latent_dims['Z2'] # modality 2 specific latent dim
        self.ds = self.latent_dims['Zs'] # shared latent dim
        self.t1 = t1
        self.t2 = t2
        self.ts = max(t1, t2)  # shared modality time dimension is the max of both
        match self.mod_type:
            case "rbf":
                self.W1 = self.rbf_mod(t1, self.d1, gamma1)
                self.W2 = self.rbf_mod(t2, self.d2, gamma2)
                self.Ws = self.rbf_mod(self.ts, self.ds, (gamma1 + gamma2) / 2)  # shared modality projection for shared modality
                self.W1 = np.concatenate((self.W1, self.Ws[:t1, :]), axis=-1)  # concatenate shared modality projection
                self.W2 = np.concatenate((self.W2, self.Ws[:t2, :]), axis=-1)  # concatenate shared modality projection
            case "random":
                self.W1 = self.random_mod(t1, self.d1)
                self.W2 = self.random_mod(t2, self.d2)
                self.Ws = self.random_mod(self.ts, self.ds)  # shared modality projection for shared modality
                self.W1 = np.concatenate((self.W1, self.Ws[:t1, :]), axis=-1)  # concatenate shared modality projection
                self.W2 = np.concatenate((self.W2, self.Ws[:t2, :]), axis=-1)  # concatenate shared modality projection
            case "identity":
                self.W1 = np.ones((t1, self.d1 + self.ds))
                self.W2 = np.ones((t2, self.d2 + self.ds))
            case _:
                raise ValueError("Unsupported modulation type")
        
        return self.W1, self.W2

    def generate_labels(self, latent_factor_t, seed: int = 0):
        """
        Converts a continuous latent factor (t_Z) into a binary categorical label (0/1)
        using a fixed, random linear projection and median thresholding.
        """
        
        # latent_factor_t is (N_samples, D_latent)
        d = latent_factor_t.shape[1]
        
        # Create a simple fixed linear projector (50D -> 1D)
        projector = nn.Linear(d, 1, bias= False)
        
        # Freeze the weights (essential for fixed, ground-truth label)
        for param in projector.parameters():
            param.requires_grad = False
        
        # Apply projection and non-linearity (ReLU makes it non-trivially related)
        score_vector = projector(torch.Tensor(latent_factor_t)).numpy().flatten() 
            
        # Thresholding based on the median (ensures a balanced 50/50 split of classes)
        midprob = np.median(score_vector)
        total_labels = (score_vector >= midprob).astype(int)
        
        return total_labels

    def normalize_data(self, data):
        # Normalize data across the last dimension
        norm_data = data / np.linalg.norm(data, axis=-1, keepdims=True)
        return norm_data

    def sample_latent_factors(self, dist: Literal["normal", "vmf"] = "normal", **kwargs):
        """
        Generates latent factors based on the specified distribution.
        Args:
            dist (str): Distribution type ('normal' -> Normal Distribution or 'vmf' -> Von Mises-Fisher Distribution). Defaults to 'normal'.
            **kwargs: Additional parameters for each distribution.
        Returns:
            data (Dict[str, np.ndarray]): Dictionary containing generated latent factors for 'Z1', 'Zs', and 'Z2'.
        """
        data = {}
        match dist:
            case "normal":
                sigmas = kwargs.get('sigmas', [1.0, 1.0, 1.0])
                for i, (k, d) in enumerate(self.latent_dims.items()):
                    data[k] = multivariate_normal(np.zeros((d,)), np.eye(d) * sigmas[i]).rvs(self.N_data)
            case "vmf":
                locs = kwargs.get('locs', np.array([torch.nn.functional.normalize(torch.randn(d), dim=0) for d in self.latent_dims.values()]))
                kappas = kwargs.get('kappas', [100.0, 100.0, 100.0])
                for i, (k, d) in enumerate(self.latent_dims.items()):
                    loc = locs[i]
                    kappa = kappas[i]
                    data[k] = vonmises_fisher(loc, kappa).rvs(self.N_data)
            case _:
                raise ValueError("Unsupported distribution type")
        return data

    def create_dataset(self, dist: Literal["normal", "vmf"] = "normal", t1: int = 5, t2: int = 5, gamma1: float = 10.0, gamma2: float = 10.0, normalize: bool = True, **kwargs):
        
        # Generate latent factors
        data = self.sample_latent_factors(dist= dist, **kwargs)
        
        t_Z1 = data['Z1']
        t_Zs = data['Zs']
        t_Z2 = data['Z2']

        if not hasattr(self, 'W1') or not hasattr(self, 'W2'):
            print("Modulation matrices not found, creating with default parameters for each modality.")
            self.create_modulation_mats(t1= t1, t2= t2, gamma1=gamma1, gamma2=gamma2)

        # generate modulation matrices
        Z1 = np.concatenate((t_Z1, t_Zs), axis=-1)  # Latent representation for modality 1
        Z2 = np.concatenate((t_Z2, t_Zs), axis=-1)  # Latent representation for modality 2


        X1 = Z1[:, None, :] * self.W1[None, :, :] # Modulated data for modality 1
        X2 = Z2[:, None, :] * self.W2[None, :, :] # Modulated data for modality 2
        
        print(f"Generated X1 sample: {X1[0, :2, :]}, X2 sample: {X2[0, :2, :]}")
        X1 = self.normalize_data(X1) if normalize else X1
        X2 = self.normalize_data(X2) if normalize else X2
        
        # --- D. Generate Disentanglement Labels (Y1, Y2, Ys) ---
    
        # Y1: Derived ONLY from t_Z1 (Specific 1)
        labels_1 = self.generate_labels(t_Z1, seed= np.random.randint(0, 10000))
        
        # Ys: Derived ONLY from t_Zs (Shared)
        labels_s = self.generate_labels(t_Zs, seed= np.random.randint(0, 10000))
        
        # Y2: Derived ONLY from t_Z2 (Specific 2)
        labels_2 = self.generate_labels(t_Z2, seed= np.random.randint(0, 10000))
        
        print(f"X1 shape: {X1.shape}, X2 shape: {X2.shape}")
        total_data = [(x1, x2) for (x1, x2) in zip(X1, X2)] # list of tuples for each sample

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
        print(f"Modality 1 data shape: {len(self.dataset_dict['total_data'])}")
        print(f"Labels 1 shape: {self.dataset_dict['labels_1'].shape}, Unique classes: {np.unique(self.dataset_dict['labels_1'])}")
        print(f"Labels 2 shape: {self.dataset_dict['labels_2'].shape}, Unique classes: {np.unique(self.dataset_dict['labels_2'])}")
        print(f"Labels s shape: {self.dataset_dict['labels_s'].shape}, Unique classes: {np.unique(self.dataset_dict['labels_s'])}")

    # Defining simple aumentations
    def noise(self, x, scale= 0.01):
        """
        Adds Gaussian noise to the input tensor.
        Args:
            x (torch.Tensor): Input data tensor.
            snr_db (float): Signal-to-noise ratio in decibels.
        Returns:
            torch.Tensor: Noisy data tensor.
        """
        epsilon = 0.01  # fraction of vector norm

        # generate random Gaussian noise
        noise = torch.normal(mean=0.0, std=1.0, size=x.shape)

        # scale noise accordingly so that it scales proportionally to each direction's magnitude
        noise = noise * x * epsilon

        noisy_x = x + noise
        return noisy_x

    def swap(self, x):
        """
        Swaps the first and second halves of the input tensor along the last dimension, i.e., each input array is swapped along the columns.
        Args:
            x (torch.Tensor): Input data tensor.
        Returns:
            torch.Tensor: Swapped data tensor.
        """
        mid = x.shape[-1] // 2
        swapped = torch.cat([x[..., mid:], x[..., :mid]], dim=-1)
        return swapped

    def random_drop(self, x, drop_scale=10):
        """
        Randomly drops a fraction of the input tensor's elements by setting them to zero.
        Args:
            x (torch.Tensor): Input data tensor.
            drop_scale (int): The fraction of elements to drop (1/ drop_scale).
        Returns:
            x_aug (torch.Tensor): Data tensor with random elements dropped.
        """
        seq_len = x.shape[-2]
        feat_dim = x.shape[-1]
        drop_num = (seq_len * feat_dim) // drop_scale # total number of elements to drop, given that the last two dims are (seq length x features)
        drop_idxs_x = np.random.choice(seq_len, drop_num // 2, replace= True)
        drop_idxs_y = np.random.choice(feat_dim, drop_num // 2, replace= True)
        x_aug = torch.clone(x)
        x_aug[..., drop_idxs_x, drop_idxs_y] = 0.0
        return x_aug


    def augment_data(self, X, aug_type: Literal['noise', 'swap', 'random_drop', 'random']='noise', **kwargs):
        """
        Simple data augmentation by adding Gaussian noise.
        Args:
            X (torch.Tensor): Input data tensor.
            aug_type (str): Type of augmentation ('noise', 'swap', 'random_drop').
            *args: Additional arguments for the augmentation function.
        Returns:
            X_aug (torch.Tensor): Augmented data tensor.
        """
        aug = aug_type if aug_type != 'random' else np.random.choice(['swap', 'random_drop'])
        match aug:
            case 'noise':
                X_aug = self.noise(X, kwargs.get("scale", 1e-3))
            case 'swap':
                X_aug = self.swap(X)    
            case 'random_drop':
                X_aug = self.random_drop(X, kwargs.get("drop_scale", 10))
            case _:
                raise ValueError(f"Unsupported augmentation type: {aug_type}")
        return X_aug

    # configure seed for all randomness
    @staticmethod   
    def set_seed(seed: int, device= 'cpu'):
        np.random.seed(seed)

        if device == "cuda":
            try:
                torch.cuda.manual_seed_all(seed)
            except RuntimeError:
                pass  # CUDA not properly initialized, continue without it



    
