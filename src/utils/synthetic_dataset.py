import torch
import numpy as np
from torch.utils.data import Dataset
# Adjust sys.path to import always from src
import os
import sys
import torch
from typing import Dict, Any, Callable, Optional, Sequence, Tuple, List, Union, Literal
import torch.nn as nn
from scipy.stats import vonmises_fisher, multivariate_normal, special_ortho_group, ortho_group
from scipy.linalg import block_diag
from itertools import combinations
from einops import rearrange
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

def save_data_split(train_dataset, test_dataset, save_path: str, split_id: str= "0"):
    os.makedirs(save_path, exist_ok=True)
    torch.save({'train_dataset': train_dataset, 'test_dataset': test_dataset}, os.path.join(save_path, f"data_split_{split_id}.pt"))
    print(f"Data split saved at {save_path}")
    return

class MultimodalDataset(Dataset):
  def __init__(self, total_data, labels_u = None, labels_s=None, t_u= None, t_s= None):
    self.data = total_data
    self.num_modalities = len(self.data[0])
    self.labels_u = labels_u if labels_u is not None else {}
    self.labels_s = labels_s if labels_s is not None else {}
    self.u_keys = sorted(self.labels_u.keys())
    self.s_keys = sorted(self.labels_s.keys())
    self.t_u = t_u if t_u is not None else {}
    self.t_s = t_s if t_s is not None else {}
    
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
            t_u = {k: self.t_u[k][idx] for k in self.u_keys}
            t_s = {k: self.t_s[k][idx] for k in self.s_keys}
            return x, u, s, t_u, t_s
            

# This class handles the generation of synthetic dataset
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

    def create_block_diag_mask(self, size: int, block_size: int):
        mask = np.zeros((size, size))
        num_blocks = size // block_size
        for b in range(num_blocks):
            start = b * block_size
            end = start + block_size
            mask[start:end, start:end] = 1
        return mask

    

    def create_dataset(self, dist: Literal["normal", "vmf"] = "normal", ts: List[int] = [5, 5], gammas: List[float] = [10.0, 10.0], normalize: bool = True, **kwargs):
        # Generate latent factors
        latent_factors = self.sample_latent_factors(dist= dist, **kwargs)
        
        # Create modality data
        Z = {}
        for i in range(self.M):
            Z[i] = self.concat_components(latent_factors, [S for S in latent_factors.keys() if i in S])
            print(f"Modality {i+1} latent factor shape: {Z[i].shape}")
    
        # Create transformation matrices
        self.create_transformation_mats(ts= ts, gammas= gammas)

        # generate transformation matrices

        X= {}
        for m in range(self.M):
            # Uncomment this block for the standard transformation matrices - this creates no correlation across time
            # X[m] = Z[m][:, None, :] * self.W[m][None, :, :]  # Modality m data across time
            # X[m] = np.einsum('t k d, n d -> n t k', self.W[m], Z[m])  # Modality m data across time if using block diagonal matrices
            # normalize
            # X[m] = self.normalize_data(X[m]) if normalize else X[m]
            z_i = Z[m][:, :]  # (N, D)
            pho1 = float(np.random.uniform(0.8, 0.95, (1,)))
            Wt_1 = np.random.uniform(-1.0, 1.0, (self.dim, self.dim))
            
            # Zero out cross terms for latents, e.g. block diagonal matrices
            mask = self.create_block_diag_mask(self.dim, self.latent_dim)
            Wt_1 *= mask
            Xi_temp = np.zeros((self.N_data, self.ts[m], self.dim))
            Xi_temp[:, 0, :] = z_i @ Wt_1.T
            
            for t in range(1, self.ts[m]):
                Wt = pho1 * Wt_1 + (1 - pho1) * np.random.uniform(-1.0, 1.0, (self.dim, self.dim))
                # Zero out cross terms for latents, e.g. block diagonal matrices
                Wt *= mask
                Xi_temp[:, t, :] = z_i @ Wt.T
                Wt_1 = Wt

            X[m] = Xi_temp  # (N, T, D)

            X[m] = self.normalize_data(X[m]) if normalize else X[m]
        
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
    def noise(x, scale= 0.01, generator= None):
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
        noise = torch.normal(mean=0.0, std=1.0, size=x.shape, generator=generator).to(x.device)

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
    def random_drop(x, drop_scale=10, generator= None):
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
        num_samples = (seq_len * feat_dim) // (2*drop_scale) # total number of elements to drop, given that the last two dims are (seq length x features)
        drop_idxs_x = torch.randint(
            seq_len, (num_samples,), generator=generator
        ).to(x.device)

        drop_idxs_y = torch.randint(
            feat_dim, (num_samples,), generator=generator
        ).to(x.device)

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
        aug = aug_type if aug_type != 'random' else np.random.choice(['noise', 'random_drop'])
        match aug:
            case 'noise':
                X_aug = GenerateData.noise(X, kwargs.get("scale", 1e-3), generator=kwargs.get("generator", None))
            case 'swap':
                X_aug = GenerateData.swap(X)    
            case 'random_drop':
                X_aug = GenerateData.random_drop(X, kwargs.get("drop_scale", 10), generator=kwargs.get("generator", None))
            case _:
                raise ValueError(f"Unsupported augmentation type: {aug_type}")
        return X_aug



# This class handles the generation of synthetic dataset in a general token-like format
class GenerateTokenizedData(GenerateData):
    """
    Inherits from GenerateData and overrides the create_dataset method to generate data in a token-like format, where each modality is represented as a sequence of token embeddings.
    """

    def gen_latent_factors_transforms(self, max_ts: int, **kwargs):
        """
        Generates the base transformation matrices for each latent factor. These transformations are shared across modalities, but 
        latent-factor specific. Each transform A_lat ~ N(0, sigma^2 * I) where sigma is a hyperparameter that controls the variance 
        of the transformation, if not provided defaults to 1.
        Args:
            max_ts (int): Maximum number of tokens (embeddings) across modalities, used to add small token-level variation.
            **kwargs: Additional parameters for each distribution.
        Returns:
            data (Dict[str, np.ndarray]): Dictionary containing generated latent factors' transformations.
        """
        latent_factors_transforms = {} # -> latent factor type transformation
        latent_factors_deltas = {} # -> token-level deltas for each latent factor, sigma_d << sigma_A
        sigma = kwargs.get('sigma', 1.0)
        sigma_d = sigma / 100.0
        for k in range(1, self.M + 1):
            for combo in combinations(range(self.M), k):
                latent_factors_transforms[frozenset(combo)] = np.random.normal(0, sigma, (self.latent_dim, self.latent_dim))
                latent_factors_deltas[frozenset(combo)] = np.random.normal(0, sigma_d, (max_ts, self.latent_dim, self.latent_dim))
        return latent_factors_transforms, latent_factors_deltas

    def gen_modality_token_masks(self, ts: List[int] = [5, 5]):
        """
        Generates masks that are modality and token-specific, so that each token (embedding) holds information about a different subset of latent factors.
        Args:
            ts (List[int]): List of token lengths for each modality, used to create token-level variation.
        Returns:
            data (Dict[str, np.ndarray]): Dictionary containing generated modality and token-specific masks.
        """
        modality_token_masks = {}
        for m in range(self.M):
            # create binary mask of shape (T_m, num_latent_factors)
            Mask_m = np.mod(np.random.permutation(ts[m] * (2**(self.M - 1))).reshape(ts[m], 2**(self.M - 1)), 2) 
        
            # check that no col is zero, so that all latent factors are represented in each modality
            zero_col = np.where(Mask_m.sum(axis=0) == 0)[0]
            for col in zero_col:
                rand_row = np.random.randint(0, ts[m])
                Mask_m[rand_row, col] = 1


            # (Optional but also check that no row is zero, so that all tokens have some latent factor information)
            zero_row = np.where(Mask_m.sum(axis=1) == 0)[0]
            for row in zero_row:
                rand_col = np.random.randint(0, 2**(self.M - 1))
                Mask_m[row, rand_col] = 1

            
            # expand mask to shape (T_m, D*num_latent_factors, D*num_latent_factors) for elementwise multiplication with the transformation matrices
            modality_token_masks[m] = np.asarray([block_diag(*[Mask_m[r, t]*np.ones((self.latent_dim, self.latent_dim)) for t in range(2**(self.M - 1))]) for r in range(ts[m])])

        return modality_token_masks

    def gen_modality_rot_mats(self):
        """
        Generates modality-specific rotation matrices Rm of shape (D*num_latent_factors, D*num_latent_factors)
        """
        R = {}
        for m in range(self.M):
            R[m] = special_ortho_group.rvs(self.latent_dim * 2**(self.M - 1)) # SO(N_components * latent_dim) rotation matrix
        return R

    def apply_nonlinearity(self, X, **kwargs):
        """
        Applies a non-linearity to the data.
        Args:
            X (np.ndarray): Input data array.
        Returns:
            np.ndarray: Non-linearly transformed data array.
        """
        alpha = kwargs.get('nonlin_alpha', 0.3)
        return np.tanh(alpha*X)  # Example non-linearity, can be replaced with others

    def create_dataset(self, dist: Literal["normal", "vmf"] = "normal", ts: List[int] = [5, 5], gammas: List[float] = [10.0, 10.0], normalize: bool = True, add_nonlinearity: bool = True, **kwargs):
        
        # Generate latent factors, transforms and masks
        latent_factors = self.sample_latent_factors(dist= dist, **kwargs)
        latent_factors_transforms, latent_factors_deltas = self.gen_latent_factors_transforms(max_ts=max(ts), **kwargs)
        modality_token_masks = self.gen_modality_token_masks(ts= ts)
        modality_rot_mats = self.gen_modality_rot_mats()

    
        # Create modality data
        Z = {} # concatenated latent factors for each modality
        A = {} # transformations for each latent factor and token, before applying modality-specific rotations
        W = {} # final transformations for each modality
        X = {} # final data for each modality
        for i in range(self.M):
            # Z = [Z_1, Z_2, Z_s]
            Z[i] = self.concat_components(latent_factors, [S for S in latent_factors.keys() if i in S])

            # At = block_diag([A_1 + latent_factors_deltas(t), A_2 + latent_factors_deltas(t), A_s + latent_factors_deltas(t)]), for all t in ts[i]
            A[i] = [block_diag(*[latent_factors_transforms[S] + latent_factors_deltas[S][t] for S in latent_factors.keys() if i in S]) for t in range(ts[i])]  
            
            
            # apply modality and token-specific masks to the transformations At for modality i, so that each 
            # token (embedding) in modality i contains a different subset of latent factor information
            A[i] = [A[i][t] * modality_token_masks[i][t] for t in range(ts[i])]
            
            # apply modality-specific rotations
            W[i] = np.asarray([modality_rot_mats[i] @ A[i][t] for t in range(ts[i])])
            
            # create data
            X[i] = np.einsum('tio,ni->nto', W[i], Z[i]) 
            print(f"Modality {i+1} data shape: {X[i].shape}")
            X[i] = self.apply_nonlinearity(X[i], **kwargs) if add_nonlinearity else X[i]
            X[i] = self.normalize_data(X[i]) if normalize else X[i]


        total_data = list(zip(*[X[m] for m in range(self.M)]))  # list of tuples for each sample
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
        aug = aug_type if aug_type != 'random' else np.random.choice(['noise', 'random_drop'])
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



# This class handles the generation of synthetic dataset in a simple embedding like format
class GeneratePermData(GenerateData):
    """
    Inherits from GenerateData and overrides the create_dataset method to generate data. In this format, each modality is represented as a sequence of embeddings, derived from a base vector with linear transformations and permutations.
    """

    def gen_latent_factors_transforms(self, max_ts: int, **kwargs):
        """
        Generates the base transformation matrices for each latent factor. These transformations are shared across modalities, but 
        latent-factor specific. Each transform A_lat ~ N(0, sigma^2 * I) where sigma is a hyperparameter that controls the variance 
        of the transformation, if not provided defaults to 1.
        Args:
            max_ts (int): Maximum number of tokens (embeddings) across modalities, used to add small token-level variation.
            **kwargs: Additional parameters for each distribution.
        Returns:
            data (Dict[str, np.ndarray]): Dictionary containing generated latent factors' transformations.
        """
        latent_factors_transforms = {} # -> latent factor type transformation
        latent_factors_deltas = {} # -> token-level deltas for each latent factor, sigma_d << sigma_A
        sigma = kwargs.get('sigma', 1.0)
        sigma_d = sigma / 100.0
        for k in range(1, self.M + 1):
            for combo in combinations(range(self.M), k):
                latent_factors_transforms[frozenset(combo)] = np.random.normal(0, sigma, (self.latent_dim, self.latent_dim))
                latent_factors_deltas[frozenset(combo)] = np.random.normal(0, sigma_d, (max_ts, self.latent_dim, self.latent_dim))
        return latent_factors_transforms, latent_factors_deltas


    def gen_modality_rot_mats(self):
        """
        Generates modality-specific rotation matrices Rm of shape (D*num_latent_factors, D*num_latent_factors)
        """
        R = {}
        for m in range(self.M):
            R[m] = special_ortho_group.rvs(self.latent_dim * 2**(self.M - 1)) # SO(N_components * latent_dim) rotation matrix
        return R

    def apply_nonlinearity(self, X, **kwargs):
        """
        Applies a non-linearity to the data.
        Args:
            X (np.ndarray): Input data array.
        Returns:
            np.ndarray: Non-linearly transformed data array.
        """
        alpha = kwargs.get('nonlin_alpha', 0.3)
        return np.tanh(alpha*X)  # Example non-linearity, can be replaced with others

    def create_dataset(self, dist: Literal["normal", "vmf"] = "normal", ts: List[int] = [5, 5], gammas: List[float] = [10.0, 10.0], normalize: bool = True, add_nonlinearity: bool = True, **kwargs):
        print(f"Creating data with dist={dist}, ts={ts}, gammas={gammas}, normalize={normalize}, add_nonlinearity={add_nonlinearity}, kwargs={kwargs}")
        # Generate latent factors, transforms and masks
        latent_factors = self.sample_latent_factors(dist= dist, **kwargs)
        latent_factors_transforms, _ = self.gen_latent_factors_transforms(max_ts=max(ts), **kwargs)

    
        # Create modality data
        Z = {} # concatenated latent factors for each modality
        X = {} # final data for each modality

        for i in range(self.M):
            Z[i] = self.concat_components(latent_factors, [S for S in latent_factors.keys() if i in S])

            # block diagonal projection to a higher dimension space, so that we can then reshare to the desired t, latent_dim shape each modality
            Z[i] = np.einsum('nd,db->nb', Z[i], block_diag(*[np.random.normal(0, 1, (self.latent_dim, self.latent_dim * ts[i])) for S in latent_factors.keys() if i in S]))  # (N, D) @ (D, B) -> (N, B)
            
            
            Z[i] = np.reshape(Z[i], (self.N_data, ts[i], self.latent_dim * 2**(self.M - 1)))  # (N, t, D)
            random_permutation = np.random.permutation(ts[i]) # the permutation is unique for each modality
            Z[i] = Z[i][:, random_permutation, :]  # shuffle embedding order 

            X[i] = self.apply_nonlinearity(Z[i], **kwargs) if add_nonlinearity else Z[i]
            X[i] = self.normalize_data(X[i]) if normalize else X[i]


        total_data = list(zip(*[X[m] for m in range(self.M)]))  # list of tuples for each sample
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
        aug = aug_type if aug_type != 'random' else np.random.choice(['noise', 'random_drop'])
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
