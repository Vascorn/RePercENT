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
        
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, 
                          bidirectional=bidirectional, batch_first=True, dropout= self.dropout)
        
        # Output projection layer
        gru_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(gru_output_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through GRU encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        Returns:
            torch.Tensor: Latent representation of shape (batch_size, latent_dim).
        '''
        # GRU forward pass
        gru_out, hidden = self.gru(x)  # gru_out: (batch_size, seq_len, hidden_dim * num_directions)
        
        # Use the last output from the sequence
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_dim * num_directions)
        
        # Project to latent dimension
        latent = self.fc(last_output)  # (batch_size, latent_dim)
        
        return latent



# Follows the JointDisenModel from the DisentangledSSL package (https://github.com/uhlerlab/DisentangledSSL) but modified to the structure of this code, i.e. the loss functions and training loop are defined outside the model class.
class JointOpt(nn.Module):
    def __init__(self, M: int = 2, sharedEncoders = None, 
                uniqueEncoders = None, 
                shared_projh= None,
                unique_projh= None,
                recon: bool= False,
                recDecoders = None,
                encoder_type: Literal["MLP", "GRU", "gMLP"] = "MLP",
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
            recon (bool): Whether to include decoders for reconstruction. Default is True.
            recDecoders: List of decoders for each modality, used if recon is True.
            encoder_type (Literal["MLP", "GRU", "gMLP"]): Type of encoder to use ("MLP", "GRU", or "gMLP"). Default is "GRU".
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
        
        # Initialize the reconstruction decoders for each modality if recon is True
        self.recon = recon
        
        if self.recon: #NOTE: reconstruction works only if the seq lengths are the same, extension should be made!!!!
            if recDecoders is not None:
                assert recDecoders is not None and len(recDecoders) == M, "recDecoders if provided must match the number of modalities M when recon is True"
                self.recDecoders = nn.ModuleList(recDecoders)
            else:
                self.recDecoders = nn.ModuleList([MLP(input_dim= self.latent_dim * 2, \
                                                hidden_dims= [self.latent_dim * 2], \
                                                latent_dim= self.latent_dim * self.seq_dim, \
                                                flatten_input= False) for _ in range(M)])
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
        else:
            raise ValueError("Cannot infer latent dimension from encoders. Please ensure that the encoders have a 'latent_dim' or 'd_model' attribute.")

        # if there are projection heads the latent dimension is determined by the output dimension of the projection heads
        if self.sharedProjh is not None:
            self.latent_dim = list(self.sharedProjh.values())[0].out_features


    def _set_seq_len(self, encoder):
        if hasattr(encoder, "seq_len") and self.latent_dim is not None: # gMLP case
            self.seq_dim = encoder.seq_len
        elif hasattr(encoder, "input_dim") and self.latent_dim is not None: # MLP, GRU case
            self.seq_dim = encoder.input_dim // self.latent_dim
        else:
            raise ValueError("Cannot infer sequence length from encoders. Please ensure that the encoders have an 'input_dim' attribute (for MLP) or 'seq_len' attribute (for gMLP).")


    def encode_modality(self, encoder, projh, x_i, mask= None):
        
        match self.encoder_type:
            case "gMLP":
                if mask is None:
                    mask = torch.ones(x_i.shape[0], x_i.shape[1], device= x_i.device)  # (B, seq_len)
                eps = 1e-8
                enc_out = encoder(x_i)
                masked_enc_out = enc_out * mask.to(dtype= enc_out.dtype).unsqueeze(-1)  # (B, seq_len, latent_dim)
                mean_pool = masked_enc_out.sum(dim=1) / mask.sum(dim= 1, keepdim= True).clamp(min= eps)
                return projh(mean_pool)
            case "GRU":
                pass
            
            case "MLP":
                return projh(encoder(x_i))
            case _:
                raise NotImplementedError(f"Masking for encoder type {self.encoder_type} not implemented yet")

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

        # If reconstruction is enabled, reconstruct each modality from its unique component and 
        # the shared component from the same or another modality with a random choice
        if self.recon:
            X_rec = torch.zeros((x[0].shape[0], self.M, self.seq_dim * self.latent_dim), device= x[0].device)
            for i in range(self.M):
                # choose one of the modalities (m included) randomly to provide the shared component for reconstruction
                j = torch.randint(0, self.M, (1,)).item()
                u_ij = U[:, i, j, :]
                s_ij = S_view[:, j, i, :]
                X_rec[:, i, :] = self.recDecoders[i](torch.cat([u_ij, s_ij], dim= -1))

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
        if self.recon:
            out["X_rec"] = X_rec
            out["X_orig"] = torch.stack(x, dim=1).flatten(start_dim= 2)  # (B, M, seq_dim * latent_dim)
        return out