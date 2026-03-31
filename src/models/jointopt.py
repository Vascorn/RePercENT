import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
import typing
from typing import Literal, List
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.DisentangledSSL.models import ProbabilisticEncoder 
from src.DisentangledSSL.losses import SupConLoss, ortho_loss
from src.DisentangledSSL.utils import ExponentialScheduler
from src.models.jointopt_2m import MLP
from itertools import permutations
ActivationName = typing.Literal['relu', 'gelu', 'sigmoid']


class GRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 1, bidirectional: bool = False, dropout: float = 0.2) -> None:
        '''
        GRU Encoder for sequential data.
        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of hidden state in GRU.
            latent_dim (int): Dimension of the output latent representation.
            num_layers (int): Number of GRU layers. Default is 1.
            bidirectional (bool): Whether to use bidirectional GRU. Default is False.
        '''
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        self.gru = nn.GRU(input_size=input_dim, 
                        hidden_size=hidden_dim, 
                        num_layers=num_layers, 
                        bidirectional=bidirectional, 
                        batch_first=True, 
                        dropout= self.dropout if num_layers > 1 else 0.0)
        
        # Output projection layer
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(gru_output_dim, latent_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        '''
        Forward pass through GRU encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            mask (torch.Tensor | None): Optional mask tensor of shape (batch_size, seq_len) where 1 indicates valid tokens and 0 indicates padding. Default is None.
        Returns:
            torch.Tensor: Latent representation of shape (batch_size, latent_dim).
        '''
        B, T, _ = x.shape

        if mask is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=x.device)
        else:
            if mask.shape != (B, T):
                raise ValueError(f"mask must have shape {(B, T)}, got {mask.shape}")
            lengths = mask.long().sum(dim=1)

        packed_x = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.gru(packed_x)

        if self.bidirectional:
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            last_hidden = hidden[-1]

        latent = self.fc(last_hidden)
        
        return latent



# Follows the JointDisenModel from the DisentangledSSL package (https://github.com/uhlerlab/DisentangledSSL) but modified to the structure of this code, i.e. the loss functions and training loop are defined outside the model class.
class JointOpt(nn.Module):
    def __init__(self, M: int = 2, sharedEncoders = None, 
                uniqueEncoders = None, 
                shared_projh= None,
                unique_projh= None,
                encoder_type: Literal["mlp", "gru", "gmlp"] = "mlp",
                vmfkappa: float= 1e3) -> None:
        '''
        JointOpt model for multi-modal representation learning with disentangled factors.
        Args:
            M (int): Number of modalities. Default is 2.
            sharedEncoders: List of encoders for each modality, responsible for extracting the shared representation.
            uniqueEncoders: List of encoders for each modality, responsible for extracting the unique representation.
            shared_projh: List of projection heads for shared encoder to ensure the output dimensions are all the same size. Mostly relevant
            for the gMLP case. In None, defaults to identity projections.
            unique_projh: List of projection heads for unique encoder to ensure the output dimensions are all the same size. Mostly relevant
            for the gMLP case. In None, defaults to identity projections.
            encoder_type (Literal["mlp", "gru", "gmlp"]): Type of encoder to use ("mlp", "gru", "gmlp"). Default is "mlp".
            vmfkappa (float): Concentration parameter for the vMF distribution in the probabilistic encoder heads. Default is 1e3.
        '''

        super().__init__()
        
        self.M = M  # Number of modalities

        # self.prob_heads = nn.ModuleList([ProbabilisticEncoder(nn.Identity(), distribution= "vmf", vmfkappa= 1e3) for _ in range(self.M)])  # Probabilistic heads for each of S_12 and S_21 - assuming only two modalities
        
        self.encoder_type = encoder_type
        self.uniqueEncoders = nn.ModuleDict() # List of M * (M - 1) - encoders for the unique component of each modality
        self.uniqueProjh = nn.ModuleDict()  # Projection heads for unique encoders to ensure output dimensions are the same, if needed (e.g. for gMLP case)
        self.sharedEncoders = nn.ModuleDict()  # List of M * (M - 1) - encoders for the shared components of each modality
        self.sharedProjh = nn.ModuleDict()  # Projection heads for shared encoders to ensure output dimensions are the same, if needed (e.g. for gMLP case)
        self.prob_heads = nn.ModuleDict()

        # save the order of (i,j) pairs for the probabilistic heads
        perm = torch.tensor(list(permutations(range(self.M), 2)), dtype=torch.long)  # 0-based

        self.register_buffer("perm_i", perm[:, 0], persistent=False)
        self.register_buffer("perm_j", perm[:, 1], persistent=False)

        for n, (i, j) in enumerate(zip(self.perm_i.tolist(), self.perm_j.tolist())):
            # define the encoders for the unique and shared components for modality i wrt modality j
            self.uniqueEncoders[f"U_{i+1}{j+1}"] = uniqueEncoders[n]
            self.sharedEncoders[f"S_{i+1}{j+1}"] = sharedEncoders[n]
            
            # define the projection heads for the unique and shared encoders to ensure output dimensions are the same, if needed (e.g. for gMLP case)
            self.uniqueProjh[f"U_{i+1}{j+1}"] = nn.Identity() if unique_projh is None else unique_projh[n]
            self.sharedProjh[f"S_{i+1}{j+1}"] = nn.Identity() if shared_projh is None else shared_projh[n]

            self.prob_heads[f"S_{i+1}{j+1}"] = ProbabilisticEncoder(nn.Identity(), distribution= "vmf", vmfkappa= vmfkappa)

        self._set_latent_dim(sharedEncoders[0])
        self._set_seq_len(sharedEncoders[0])

        print(f"Model initialized with latent dimension: {self.latent_dim} and sequence dimension: {self.seq_dim}")
        self.norm = lambda x: nn.functional.normalize(x, dim=-1)
        
        # indices for all unordered pairs i<j (0-based)
        idx = torch.triu_indices(self.M, self.M, offset=1)  # (2, P)
        self.register_buffer("pair_i", idx[0])  # (P,)
        self.register_buffer("pair_j", idx[1])  # (P,)
        self.P = idx.shape[1]

    def _set_latent_dim(self, encoder):
        if hasattr(encoder, "latent_dim"): # MLP, GRU case
            self.latent_dim = encoder.latent_dim

        elif hasattr(encoder, "d_model"): # gMLP case
            self.latent_dim = encoder.d_model
            # if there are projection heads the latent dimension is determined by the output dimension of the projection heads
            if list(self.sharedProjh.values())[0] is not nn.Identity():
                self.latent_dim = list(self.sharedProjh.values())[0].out_features
        else:
            raise ValueError("Cannot infer latent dimension from encoders. Please ensure that the encoders have a 'latent_dim' or 'd_model' attribute.")

        


    def _set_seq_len(self, encoder):
        if hasattr(encoder, "seq_len") and self.latent_dim is not None: # gMLP case
            self.seq_dim = encoder.seq_len
        elif hasattr(encoder, "input_dim") and self.latent_dim is not None: # MLP, GRU case
            self.seq_dim = encoder.input_dim // self.latent_dim
        else:
            raise ValueError("Cannot infer sequence length from encoders. Please ensure that the encoders have an 'input_dim' attribute (for MLP) or 'seq_len' attribute (for gMLP).")


    def encode_modality(self, encoder, projh, x_i, mask= None):
        
        match self.encoder_type:
            case "gmlp":
                if mask is None:
                    mask = torch.ones(x_i.shape[0], x_i.shape[1], device= x_i.device)  # (B, seq_len)
                eps = 1e-8
                enc_out = encoder(x_i)
                masked_enc_out = enc_out * mask.to(dtype= enc_out.dtype).unsqueeze(-1)  # (B, seq_len, latent_dim)
                mean_pool = masked_enc_out.sum(dim=1) / mask.sum(dim= 1, keepdim= True).clamp(min= eps)
                return projh(mean_pool)
            case "gru":
                return projh(encoder(x_i, mask= mask))
            
            case "mlp":
                return projh(encoder(x_i))
            case _:
                raise NotImplementedError(f"encoder type {self.encoder_type} not implemented yet")

    def forward(self, x, mask= None):
        """
        Forward pass through the original JointOpt model that uses one decoder per disentangled component.
        Args:
        x: List of input data for each modality. Length of the list should be M.
        mask: Optional list of masks for each modality, if applicable. Default is None. If the embeddings are variable length and 
            require masking, this should be taken into account, depeding on the encoder type.
        """

        assert len(x) == self.M, "Input list length must match number of modalities M"

        # extract all components and store in arrays
        # Each U[*, i, j, *] corresponds to the unique component from modality i wrt modality j, similarly for S, S_prob
        U = torch.zeros((x[0].shape[0], self.M, self.M, self.latent_dim), device= x[0].device)  # Unique components
        S_view = torch.zeros((x[0].shape[0], self.M, self.M, self.latent_dim), device= x[0].device)  # Shared components from encoders
        S_prob = torch.zeros((x[0].shape[0], self.M, self.M, self.latent_dim), device= x[0].device)  # Initialize tensor to hold probabilistic shared components

        for n, (i, j) in enumerate(zip(self.perm_i.tolist(), self.perm_j.tolist())):
            
            u_ij = self.encode_modality(self.uniqueEncoders[f"U_{i+1}{j+1}"], self.uniqueProjh[f"U_{i+1}{j+1}"], x[i], mask[i] if mask is not None else None)  # Unique component from modality i wrt modality j
            s_ij = self.encode_modality(self.sharedEncoders[f"S_{i+1}{j+1}"], self.sharedProjh[f"S_{i+1}{j+1}"], x[i], mask[i] if mask is not None else None)  # Shared component from modality i wrt modality j
            
            # add probabilistic heads for shared components
            p_s_ij_given_xi, _ = self.prob_heads[f"S_{i+1}{j+1}"](s_ij)

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