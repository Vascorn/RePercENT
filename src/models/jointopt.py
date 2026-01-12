import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
import typing
from typing import Literal, List
from src.DisentangledSSL.models import ProbabilisticEncoder 
from src.DisentangledSSL.losses import SupConLoss, ortho_loss
from src.DisentangledSSL.utils import ExponentialScheduler
from src.models.repercent_2m import simpleEncoder
from itertools import permutations


# Follows the JointDisenModel from the DisentangledSSL package (https://github.com/uhlerlab/DisentangledSSL) but modified to the structure of this code, i.e. the loss functions and training loop are defined outside the model class.
class JointOpt(nn.Module):
    def __init__(self, M: int = 2, sharedEncoders: List[simpleEncoder] = None, uniqueEncoders: List[simpleEncoder] = None, vmfkappa: float= 1e3, add_shared= False) -> None:
        '''
        JointOpt model for multi-modal representation learning with disentangled factors.
        Args:
            M (int): Number of modalities. Default is 2.
            sharedEncoders (List[simpleEncoder]): List of MLP encoders for each modality, responsible for extracting the shared representation.
            uniqueEncoders (List[simpleEncoder]): List of MLP encoders for each modality, responsible for extracting the unique representation.
            vmfkappa (float): Concentration parameter for the vMF distribution in the probabilistic encoder heads. Default is 1e3.
            add_shared (bool): Whether to add as input the extracted shared components to the unique encoders. Default is False.
        '''

        super().__init__()
        
        self.M = M  # Number of modalities

        # self.prob_heads = nn.ModuleList([ProbabilisticEncoder(nn.Identity(), distribution= "vmf", vmfkappa= 1e3) for _ in range(self.M)])  # Probabilistic heads for each of S_12 and S_21 - assuming only two modalities
        

        self.uniqueEncoders = nn.ModuleDict() # List of M * (M - 1) - MLP encoders for the unique component of each modality
        self.sharedEncoders = nn.ModuleDict()  # List of M * (M - 1) - MLP encoders for the shared components of each modality
        self.prob_heads = nn.ModuleList()

        # save the order of (i,j) pairs for the probabilistic heads
        perm = torch.tensor(list(permutations(range(self.M), 2)), dtype=torch.long)  # 0-based
        self.register_buffer("perm_i", perm[:, 0], persistent=False)
        self.register_buffer("perm_j", perm[:, 1], persistent=False)

        for i, j in zip(self.perm_i, self.perm_j):
            self.prob_heads.append(ProbabilisticEncoder(nn.Identity(), distribution= "vmf", vmfkappa= vmfkappa))

        for n, (i, j) in enumerate(zip(self.perm_i.tolist(), self.perm_j.tolist())):
            self.uniqueEncoders[f"U_{i+1}{j+1}"] = uniqueEncoders[n]
            self.sharedEncoders[f"S_{i+1}{j+1}"] = sharedEncoders[n]


        self.latent_dim = uniqueEncoders[0].latent_dim  # assuming all encoders have the same latent dim
        self.norm = lambda x: nn.functional.normalize(x, dim=-1)
        self.add_shared = add_shared
        
        # indices for all unordered pairs i<j (0-based)
        idx = torch.triu_indices(self.M, self.M, offset=1)  # (2, P)
        self.register_buffer("pair_i", idx[0])  # (P,)
        self.register_buffer("pair_j", idx[1])  # (P,)
        self.P = idx.shape[1]

    def forward(self, x):
        """
        Forward pass through the original JointOpt model that uses one decoder per disentangled component.
        Args:
        x: List of input data for each modality. Length of the list should be M.
        """

        assert len(x) == self.M, "Input list length must match number of modalities M"

        # extract all components and store in arrays
        # Each U[*, i, j, *] corresponds to the unique component from modality i wrt modality j, similarly for S, S_prob
        U = torch.zeros((x[0].shape[0], self.M, self.M, self.latent_dim), device= x[0].device)  # Unique components
        S_view = torch.zeros((x[0].shape[0], self.M, self.M, self.latent_dim), device= x[0].device)  # Shared components from encoders
        S_prob = torch.zeros((x[0].shape[0], self.M, self.M, self.latent_dim), device= x[0].device)  # Initialize tensor to hold probabilistic shared components

        for n, (i, j) in enumerate(zip(self.perm_i.tolist(), self.perm_j.tolist())):
            
            u_ij = self.uniqueEncoders[f"U_{i+1}{j+1}"](x[i])  # Unique component from modality i wrt modality j
            s_ij = self.sharedEncoders[f"S_{i+1}{j+1}"](x[i])  # Shared component from modality i wrt modality j
            
            # add probabilistic heads for shared components
            p_s_ij_given_xi, _ = self.prob_heads[n](s_ij)

            s_ij_prob= p_s_ij_given_xi.rsample()

            U[:, i, j, :] = u_ij
            S_view[:, i, j, :] = s_ij
            S_prob[:, i, j, :] = s_ij_prob

        # --- S_concat: (B, P, 2, D) = [s_ij, s_ji] ---
        i = self.pair_i
        j = self.pair_j
        S_concat = torch.stack([S_prob[:, i, j, :], S_prob[:, j, i, :]], dim=2)  # (B,P,2,D)
        S_concat = self.norm(S_concat)


        # --- Z_concat: (B, P, 2, 2D) ---
        # view 0 for pair (i,j): [u_ij, s_ji]
        Z_i_concat = torch.cat([U[:, i, j, :], S_prob[:, j, i, :]], dim=-1)  # (B,P,2D)
        Z_i_concat = self.norm(Z_i_concat)
        # view 1 for pair (i,j): [u_ji, s_ij]
        Z_j_concat = torch.cat([U[:, j, i, :], S_prob[:, i, j, :]], dim=-1)  # (B,P,2D)
        Z_j_concat = self.norm(Z_j_concat)


        out = {"U": U, "S_view": S_view, "S_prob": S_prob, "S_concat": S_concat, "Z_i_concat": Z_i_concat, "Z_j_concat": Z_j_concat}
        return out