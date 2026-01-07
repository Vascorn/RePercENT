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
from itertools import combinations
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
  def __init__(self, total_data, labels_u = None, labels_s=None):
    self.data = total_data
    self.num_modalities = len(self.data[0])
    self.labels_u = labels_u if labels_u is not None else {}
    self.labels_s = labels_s if labels_s is not None else {}
    self.u_keys = sorted(self.labels_u.keys())
    self.s_keys = sorted(self.labels_s.keys())
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
        x = tuple([torch.from_numpy(self.data[idx][m]).to(torch.float32) for m in range(self.num_modalities)])

        if not self.labels_u and not self.labels_s:
            return x
        if not self.labels_s and not self.labels_u:
            return x
        else:
            u = {k: self.labels_u[k][idx] for k in self.u_keys}
            s = {k: self.labels_s[k][idx] for k in self.s_keys}
            return x, u, s
            
  def sample_batch(self, batch_size):
    sample_idxs = np.random.choice(self.__len__(), batch_size, replace=False)
    samples = self.__getitem__(sample_idxs)
    return samples

# This class handles the generation of synthetic dataset for the two-modality case
class GenerateData():
    def __init__(self, N_data: int, trans_type: str= "uniform", latent_dim: int= 16, M: int= 2):
        """
        Args:
            N_data (int): Number of data samples to generate.
            trans_type (str): Type of transformation ('rbf', 'random', 'identity'). This defines how each modality vectors with be projected across time.
            latent_dim (int): Dictionary specifying the dimensions of latent factor. For M modalities, each modality can be decomposed in 2^{M -1} components and there will be in total
                            2^M - 1 latent factors. E.g. for M=2, |X1| = |u_1,2| + |s_12| = 2 * latent_dim, |X2| = |u_2,1| + |s_12| = 2 * latent_dim. For M = 3, 
                            |X1| = |u_1,23| + |u_12,3| + |u_13,2| + |s_123| = 4 * latent_dim, etc.
            M (int): Number of modalities.
        """
        self.N_data = N_data
        self.trans_type = trans_type
        self.latent_dim = latent_dim
        self.M = M

    def create_transformation_mats(self, ts: List[int] = [5, 5], gammas: List[float] = [10.0, 10.0]):
        self.dim = (2**(self.M - 1)) * self.latent_dim  # total latent dim across all factors for each modality
        self.ts = ts

        match self.trans_type:
            case "uniform":
                self.W = {}
                for i in range(self.M):
                    t = ts[i]
                    self.W[i] = np.random.uniform(-1.0, 1.0, (t, self.dim))
                # Uncomment for the block diagonal variant and ajdust the multiplication accordingly in the create_dataset method
                # self.W = {}
                # for i in range(self.M):
                #     t = ts[i]
                #     self.W[i] = np.random.uniform(-1, 1, (t, self.dim, self.dim))
                #     # Zero out cross terms for latents, e.g. block diagonal matrices
                #     for j in range(2**self.M - 1):
                #         for k in range(2**self.M - 1):
                #             if j != k:
                #                 self.W[i][:, j * self.latent_dim:(j + 1) * self.latent_dim, k * self.latent_dim:(k + 1) * self.latent_dim] = 0.0
            case "normal":
                self.W = {}
                for i in range(self.M):
                    t = ts[i]
                    self.W[i] = np.random.normal(0, 1, (t, self.dim))

                # Uncomment for the block diagonal variant and adjust the multiplication accordingly in the create_dataset method
                # self.W = {}
                # for i in range(self.M):
                #     t = ts[i]
                #     self.W[i] = np.random.normal(0, 1, (t, self.dim, self.dim))
                #     # Zero out cross terms for latents, e.g. block diagonal matrices
                #     for j in range(2**self.M - 1):
                #         for k in range(2**self.M - 1):
                #             if j != k:
                #                 self.W[i][:, j * self.latent_dim:(j + 1) * self.latent_dim, k * self.latent_dim:(k + 1) * self.latent_dim] = 0.0
            case "identity":
                self.W = {}
                for i in range(self.M):
                    t = ts[i]
                    self.W[i] = np.ones((t, self.dim))
                # Uncomment for the block diagonal variant and adjust the multiplication accordingly in the create_dataset method
                # self.W = {}
                # for i in range(self.M):
                #     t = ts[i]
                #     self.W[i] = np.ones((t, self.dim, self.dim))
                # Zero out cross terms for latents, e.g. block diagonal matrices
                #     for j in range(2**self.M - 1):
                #         for k in range(2**self.M - 1):
                #             if j != k:
                #                 self.W[i][:, j * self.latent_dim:(j + 1) * self.latent_dim, k * self.latent_dim:(k + 1) * self.latent_dim] = 0.0
            case _:
                raise ValueError("Unsupported modulation type")
        
        return self.W

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

    def normalize_data(self, data, eps: float = 1e-8):
        # Normalize data across the last dimension
        norm_data = data / (np.linalg.norm(data, axis=-1, keepdims=True) + eps)
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
        latent_factors = {}
        match dist:
            case "normal":
                sigma = kwargs.get('sigma', 1.0)
                for k in range(1, self.M + 1):
                    for combo in combinations(range(self.M), k):
                        latent_factors[frozenset(combo)] = multivariate_normal(np.zeros((self.latent_dim, )), np.eye(self.latent_dim) * sigma).rvs(self.N_data)
            case "vmf":
                locs = kwargs.get('locs', np.array([torch.nn.functional.normalize(torch.randn(self.latent_dim), dim=0) for _ in range(2**self.M - 1)]))
                kappas = kwargs.get('kappas', [100.0] * (2**self.M - 1))
                idx = 0
                for k in range(1, self.M + 1):
                    for combo in combinations(range(self.M), k):
                        loc = locs[idx]
                        kappa = kappas[idx]
                        latent_factors[frozenset(combo)] = vonmises_fisher(loc, kappa).rvs(self.N_data)
                        idx += 1
            case _:
                raise ValueError("Unsupported distribution type")
        return latent_factors


    def concat_components(self, comps, subsets):
        return np.concatenate([comps[S] for S in subsets], axis=-1)
    
    def subsets_pair_shared(self, latent_factors, i, j):
        """Return the list of subsets that contain both i and j.
        Args:
            latent_factors (Dict[frozenset, np.ndarray]): Dictionary of latent factors.
            i (int): Index of the first modality.
            j (int): Index of the second modality.
        """
        return [S for S in latent_factors.keys() if (i in S and j in S)]


    def subsets_unique_wrt(self, latent_factors, i, j):
        """Return the list of subsets that contain i but not j.
        Args:
            latent_factors (Dict[frozenset, np.ndarray]): Dictionary of latent factors.
            i (int): Index of the modality to include.
            j (int): Index of the modality to exclude.
        """
        return [S for S in latent_factors.keys() if (i in S and j not in S)]

    def create_dataset(self, dist: Literal["normal", "vmf"] = "normal", ts: List[int] = [5, 5], gammas: List[float] = [10.0, 10.0], normalize: bool = True, **kwargs):
        
        # Generate latent factors
        latent_factors = self.sample_latent_factors(dist= dist, **kwargs)

        # Create modality data
        Z = {}
        for i in range(self.M):
            Z[i] = self.concat_components(latent_factors, [S for S in latent_factors.keys() if i in S])
    
        # Create transformation matrices
        self.create_transformation_mats(ts= ts, gammas= gammas)

        # generate transformation matrices

        X= {}
        for i in range(self.M):
            X[i] = Z[i][:, None, :] * self.W[i][None, :, :]  # Modality i data across time
            # X[i] = np.einsum('t k d, n d -> n t k', self.W[i], Z[i])  # Modality i data across time if using block diagonal matrices
            X[i] = self.normalize_data(X[i]) if normalize else X[i]
        
        total_data = list(zip(*[X[m] for m in range(self.M)]))  # list of tuples for each sample
        # Generate the labels for the shared and unique components
        
        t_u = {}
        t_s = {}
        labels_u = {}
        labels_s = {}
        for i in range(self.M):
            for j in range(self.M):
                if i != j:
                    # Unique components u_{i,j} and labels
                    key = f'u_{i+1}{j+1}'
                    subsets_u = self.subsets_unique_wrt(latent_factors, i, j)
                    temp_latent_u = self.concat_components(latent_factors, subsets_u)
                    t_u[key] = temp_latent_u
                    labels_u[key] = self.generate_labels(temp_latent_u, seed= np.random.randint(0, 10000))
                if i < j:
                    # Shared components s_{i,j} and labels
                    key_s = f's_{i+1}{j+1}'
                    subsets_s = self.subsets_pair_shared(latent_factors, i, j)
                    temp_latent_s = self.concat_components(latent_factors, subsets_s)
                    t_s[key_s] = temp_latent_s
                    labels_s[key_s] = self.generate_labels(temp_latent_s, seed= np.random.randint(0, 10000))

        # pack into a dictionary
        self.dataset_dict = {
            'total_data': total_data,
            'labels_u': labels_u,
            'labels_s': labels_s,
            't_u': t_u,
            't_s': t_s
        }
        return self.dataset_dict

    def print_dataset_info(self):
        if not hasattr(self, 'dataset_dict'):
            raise ValueError("Dataset not created yet. Please run create_dataset() first.")
        
        print("Dataset Information:")
        print(f"Number of samples: {self.N_data}")
        print(f"Number of modalities: {self.M}")
        for i in range(self.M):
            print(f"Modality {i+1} data shape: {self.dataset_dict['total_data'][0][i].shape}")
        for key, value in self.dataset_dict['labels_u'].items():
            print(f"Labels for unique component {key} shape: {value.shape}")
        for key, value in self.dataset_dict['labels_s'].items():
            print(f"Labels for shared component {key} shape: {value.shape}")
        for key, value in self.dataset_dict['t_u'].items():
            print(f"Latent factors for unique component {key} shape: {value.shape}")
        for key, value in self.dataset_dict['t_s'].items():
            print(f"Latent factors for shared component {key} shape: {value.shape}")

    # Defining simple aumentations
    @staticmethod
    def noise(x, scale= 0.01):
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

    @staticmethod
    def swap(x):
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
    @staticmethod
    def random_drop(x, drop_scale=10):
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

    @staticmethod
    def augment_data(X, aug_type: Literal['noise', 'swap', 'random_drop', 'random']='noise', **kwargs):
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
                X_aug = GenerateData.noise(X, kwargs.get("scale", 1e-3))
            case 'swap':
                X_aug = GenerateData.swap(X)    
            case 'random_drop':
                X_aug = GenerateData.random_drop(X, kwargs.get("drop_scale", 10))
            case _:
                raise ValueError(f"Unsupported augmentation type: {aug_type}")
        return X_aug


# # This class handles the generation of synthetic dataset for the two-modality case
# class GenerateData():
#     def __init__(self, N_data: int, trans_type: str= "uniform", latent_dims: Dict[str, int]= {'Zs': 50, 'Z1': 50, 'Z2': 50}):
#         """
#         Args:
#             N_data (int): Number of data samples to generate.
#             trans_type (str): Type of transformation ('rbf', 'random', 'identity'). This defines how each modality vectors with be projected across time.
#             latent_dims (Dict[str, int]): Dictionary specifying the dimensions of shared and modality-specific latent factors.
#         """
#         self.N_data = N_data
#         self.trans_type = trans_type
#         self.latent_dims = latent_dims
        

#     def create_transformation_mats(self, t1: int = 5, t2: int = 5, gamma1: float = 10.0, gamma2: float = 10.0):
#         self.d1 = self.latent_dims['Z1'] + self.latent_dims['Zs'] # modality 1 specific + shared latent dim
#         self.d2 = self.latent_dims['Z2'] + self.latent_dims['Zs'] # modality 2 specific + sharedlatent dim
#         self.ds = self.latent_dims['Zs'] # shared latent dim
#         self.t1 = t1
#         self.t2 = t2

#         match self.trans_type:
#             case "uniform":
#                 self.W1 = np.random.uniform(-1.0, 1.0, (t1, self.d1))
#                 self.W2 = np.random.uniform(-1.0, 1.0, (t2, self.d2))
#                 # Uncomment for the block diagonal variant and ajdust the multiplication accordingly in the create_dataset method
#                 # self.W1 = np.random.uniform(-1, 1, (t1, self.d1, self.d1))
#                 # self.W1[:, self.latent_dims['Z1']:, :self.latent_dims['Z1']] = 0.0  # Zero out cross terms for shared latents
#                 # self.W1[:, :self.latent_dims['Z1'], self.latent_dims['Z1']:] = 0.0  # Zero out cross terms for shared latents
#                 # self.W2 = np.random.uniform(-1, 1, (t2, self.d2, self.d2))
#                 # self.W2[:, self.latent_dims['Z2']:, :self.latent_dims['Z2']] = 0.0  # Zero out cross terms for shared latents
#                 # self.W2[:, :self.latent_dims['Z2'], self.latent_dims['Z2']:] = 0.0  # Zero out cross terms for shared latents
#             case "random":
#                 self.W1 = np.random.normal(0, 1, (t1, self.d1))
#                 self.W2 = np.random.normal(0, 1, (t2, self.d2))
#                 # Uncomment for the block diagonal variant and adjust the multiplication accordingly in the create_dataset method
#                 # self.W1 = np.random.normal(0, 1, (t1, self.d1, self.d1))
#                 # self.W1[:, self.latent_dims['Z1']:, :self.latent_dims['Z1']] = 0.0  # Zero out cross terms for shared latents
#                 # self.W1[:, :self.latent_dims['Z1'], self.latent_dims['Z1']:] = 0.0  # Zero out cross terms for shared latents
#                 # self.W2 = np.random.normal(0, 1, (t2, self.d2, self.d2))#np.random.normal(0, 1, (t2, self.d2, self.d2))
#                 # self.W2[:, self.latent_dims['Z2']:, :self.latent_dims['Z2']] = 0.0  # Zero out cross terms for shared latents
#                 # self.W2[:, :self.latent_dims['Z2'], self.latent_dims['Z2']:] = 0.0  # Zero out cross terms for shared latents
#             case "identity":
#                 self.W1 = np.ones((t1, self.d1, self.d1))
#                 self.W2 = np.ones((t2, self.d2, self.d2))
#             case _:
#                 raise ValueError("Unsupported modulation type")
        
#         return self.W1, self.W2

#     def generate_labels(self, latent_factor_t, seed: int = 0):
#         """
#         Converts a continuous latent factor (t_Z) into a binary categorical label (0/1)
#         using a fixed, random linear projection and median thresholding.
#         """
        
#         # latent_factor_t is (N_samples, D_latent)
#         d = latent_factor_t.shape[1]
        
#         # Create a simple fixed linear projector (50D -> 1D)
#         projector = nn.Linear(d, 1, bias= False)
        
#         # Freeze the weights (essential for fixed, ground-truth label)
#         for param in projector.parameters():
#             param.requires_grad = False
        
#         # Apply projection and non-linearity (ReLU makes it non-trivially related)
#         score_vector = projector(torch.Tensor(latent_factor_t)).numpy().flatten() 
            
#         # Thresholding based on the median (ensures a balanced 50/50 split of classes)
#         midprob = np.median(score_vector)
#         total_labels = (score_vector >= midprob).astype(int)
        
#         return total_labels

#     def normalize_data(self, data, eps: float = 1e-8):
#         # Normalize data across the last dimension
#         norm_data = data / (np.linalg.norm(data, axis=-1, keepdims=True) + eps)
#         return norm_data

#     def sample_latent_factors(self, dist: Literal["normal", "vmf"] = "normal", **kwargs):
#         """
#         Generates latent factors based on the specified distribution.
#         Args:
#             dist (str): Distribution type ('normal' -> Normal Distribution or 'vmf' -> Von Mises-Fisher Distribution). Defaults to 'normal'.
#             **kwargs: Additional parameters for each distribution.
#         Returns:
#             data (Dict[str, np.ndarray]): Dictionary containing generated latent factors for 'Z1', 'Zs', and 'Z2'.
#         """
#         data = {}
#         match dist:
#             case "normal":
#                 sigmas = kwargs.get('sigmas', [1.0, 1.0, 1.0])
#                 for i, (k, d) in enumerate(self.latent_dims.items()):
#                     data[k] = multivariate_normal(np.zeros((d,)), np.eye(d) * sigmas[i]).rvs(self.N_data)
#             case "vmf":
#                 locs = kwargs.get('locs', np.array([torch.nn.functional.normalize(torch.randn(d), dim=0) for d in self.latent_dims.values()]))
#                 kappas = kwargs.get('kappas', [100.0, 100.0, 100.0])
#                 for i, (k, d) in enumerate(self.latent_dims.items()):
#                     loc = locs[i]
#                     kappa = kappas[i]
#                     data[k] = vonmises_fisher(loc, kappa).rvs(self.N_data)
#             case _:
#                 raise ValueError("Unsupported distribution type")
#         return data

#     def create_dataset(self, dist: Literal["normal", "vmf"] = "normal", t1: int = 5, t2: int = 5, gamma1: float = 10.0, gamma2: float = 10.0, normalize: bool = True, **kwargs):
        
#         # Generate latent factors
#         data = self.sample_latent_factors(dist= dist, **kwargs)
        
#         t_Z1 = data['Z1']
#         t_Zs = data['Zs']
#         t_Z2 = data['Z2']

#         if not hasattr(self, 'W1') or not hasattr(self, 'W2'):
#             print("Transformation matrices not found, creating with default parameters for each modality.")
#             self.create_transformation_mats(t1= t1, t2= t2, gamma1=gamma1, gamma2=gamma2)

#         # generate transformation matrices
#         Z1 = np.concatenate((t_Z1, t_Zs), axis=-1)  # Latent representation for modality 1
#         Z2 = np.concatenate((t_Z2, t_Zs), axis=-1)  # Latent representation for modality 2
        
#         X1 = Z1[:, None, :] * self.W1[None, :, :]  # Modality 1 data across time
#         X2 = Z2[:, None, :] * self.W2[None, :, :]  # Modality 2 data across time
#         # X1 = np.einsum('t k d, n d -> n t k', self.W1, Z1)  # Modality 1 data across time
#         # X2 = np.einsum('t k d, n d -> n t k', self.W2, Z2)  # Modality 2 data across time

#         X1 = self.normalize_data(X1) if normalize else X1
#         X2 = self.normalize_data(X2) if normalize else X2
        
#         # --- D. Generate Disentanglement Labels (Y1, Y2, Ys) ---
    
#         # Y1: Derived ONLY from t_Z1 (Specific 1)
#         labels_1 = self.generate_labels(t_Z1, seed= np.random.randint(0, 10000))
        
#         # Ys: Derived ONLY from t_Zs (Shared)
#         labels_s = self.generate_labels(t_Zs, seed= np.random.randint(0, 10000))
        
#         # Y2: Derived ONLY from t_Z2 (Specific 2)
#         labels_2 = self.generate_labels(t_Z2, seed= np.random.randint(0, 10000))
        
#         print(f"X1 shape: {X1.shape}, X2 shape: {X2.shape}")
#         total_data = [(x1, x2) for (x1, x2) in zip(X1, X2)] # list of tuples for each sample

#         # pack into a dictionary
#         self.dataset_dict = {
#             'total_data': total_data,
#             'labels_1': labels_1,
#             'labels_2': labels_2,
#             'labels_s': labels_s,
#             't_Z1': t_Z1,
#             't_Zs': t_Zs,
#             't_Z2': t_Z2
#         }
#         return self.dataset_dict

#     def print_dataset_info(self):
#         if not hasattr(self, 'dataset_dict'):
#             raise ValueError("Dataset not created yet. Please run create_dataset() first.")
        
#         print("Dataset Information:")
#         print(f"Number of samples: {self.N_data}")
#         print(f"Modality 1 data shape: {len(self.dataset_dict['total_data'])}")
#         print(f"Labels 1 shape: {self.dataset_dict['labels_1'].shape}, Unique classes: {np.unique(self.dataset_dict['labels_1'])}")
#         print(f"Labels 2 shape: {self.dataset_dict['labels_2'].shape}, Unique classes: {np.unique(self.dataset_dict['labels_2'])}")
#         print(f"Labels s shape: {self.dataset_dict['labels_s'].shape}, Unique classes: {np.unique(self.dataset_dict['labels_s'])}")

#     # Defining simple aumentations
#     @staticmethod
#     def noise(x, scale= 0.01):
#         """
#         Adds Gaussian noise to the input tensor.
#         Args:
#             x (torch.Tensor): Input data tensor.
#             snr_db (float): Signal-to-noise ratio in decibels.
#         Returns:
#             torch.Tensor: Noisy data tensor.
#         """

#         # generate random Gaussian noise
#         noise = torch.normal(mean=0.0, std=1.0, size=x.shape)

#         # scale noise accordingly so that it scales proportionally to each direction's magnitude
#         noise = noise * x * scale

#         noisy_x = x + noise
#         return noisy_x

#     @staticmethod
#     def swap(x):
#         """
#         Swaps the first and second halves of the input tensor along the last dimension, i.e., each input array is swapped along the columns.
#         Args:
#             x (torch.Tensor): Input data tensor.
#         Returns:
#             torch.Tensor: Swapped data tensor.
#         """
#         mid = x.shape[-1] // 2
#         swapped = torch.cat([x[..., mid:], x[..., :mid]], dim=-1)
#         return swapped
#     @staticmethod
#     def random_drop(x, drop_scale=10):
#         """
#         Randomly drops a fraction of the input tensor's elements by setting them to zero.
#         Args:
#             x (torch.Tensor): Input data tensor.
#             drop_scale (int): The fraction of elements to drop (1/ drop_scale).
#         Returns:
#             x_aug (torch.Tensor): Data tensor with random elements dropped.
#         """
#         seq_len = x.shape[-2]
#         feat_dim = x.shape[-1]
#         drop_num = (seq_len * feat_dim) // drop_scale # total number of elements to drop, given that the last two dims are (seq length x features)
#         drop_idxs_x = np.random.choice(seq_len, drop_num // 2, replace= True)
#         drop_idxs_y = np.random.choice(feat_dim, drop_num // 2, replace= True)
#         x_aug = torch.clone(x)
#         x_aug[..., drop_idxs_x, drop_idxs_y] = 0.0
#         return x_aug

#     @staticmethod
#     def augment_data(X, aug_type: Literal['noise', 'swap', 'random_drop', 'random']='noise', **kwargs):
#         """
#         Simple data augmentation by adding Gaussian noise.
#         Args:
#             X (torch.Tensor): Input data tensor.
#             aug_type (str): Type of augmentation ('noise', 'swap', 'random_drop').
#             *args: Additional arguments for the augmentation function.
#         Returns:
#             X_aug (torch.Tensor): Augmented data tensor.
#         """
#         aug = aug_type if aug_type != 'random' else np.random.choice(['swap', 'random_drop'])
#         match aug:
#             case 'noise':
#                 X_aug = GenerateData.noise(X, kwargs.get("scale", 1e-3))
#             case 'swap':
#                 X_aug = GenerateData.swap(X)    
#             case 'random_drop':
#                 X_aug = GenerateData.random_drop(X, kwargs.get("drop_scale", 10))
#             case _:
#                 raise ValueError(f"Unsupported augmentation type: {aug_type}")
#         return X_aug

#     # configure seed for all randomness
#     @staticmethod   
#     def set_seed(seed: int, device= 'cpu'):
#         np.random.seed(seed)

#         if device == "cuda":
#             try:
#                 torch.cuda.manual_seed_all(seed)
#             except RuntimeError:
#                 pass  # CUDA not properly initialized, continue without it



    
