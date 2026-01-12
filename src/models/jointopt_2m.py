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

ActivationName = typing.Literal['relu', 'gelu', 'sigmoid']

class simpleEncoder(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dims: List[int] = [64], latent_dim: int = 32, activation: ActivationName = 'relu', dropout: float = 0.3) -> None:
        
        super(simpleEncoder, self).__init__()
        self.input_dim = input_dim # initial input dimension
        self.hidden_dims = hidden_dims # hidden layer dimension
        self.latent_dim = latent_dim # final output dimension
        self.dropout = dropout

        match activation:
            case 'relu':
                self.activation = nn.ReLU()
            case 'gelu':
                self.activation = nn.GELU()
            case 'sigmoid':
                self.activation = nn.Sigmoid()
            case _:
                exit ("Unsupported activation function")

        prev_dim = self.input_dim
        layers: list[nn.Module] = []
        
        for h in self.hidden_dims:
            layer = nn.Linear(prev_dim, h)
            
            layers.append(layer)
            layers.append(self.activation)
            if self.dropout > 0:
                layers.append(nn.Dropout(p= self.dropout))
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, self.latent_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # flatten input except batch dimension
        x = x.flatten(start_dim= -2)
        out = self.mlp(x)
        return out

    def forward(self, x):
        # flatten input except batch dimension
        x = x.flatten(start_dim= -2)
        out = self.mlp(x)
        return out


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

        assert M == len(sharedEncoders), "Number of modalities M must match the length of disenEncoder list"
        
        self.M = M  # Number of modalities

        self.sharedEncoders = nn.ModuleList(sharedEncoders)  # List of 2 - MLP encoders for the shared components of each modality
        self.prob_heads = nn.ModuleList([ProbabilisticEncoder(nn.Identity(), distribution= "vmf", vmfkappa= 1e3) for _ in range(self.M)])  # Probabilistic heads for each of S_12 and S_21 - assuming only two modalities

        self.uniqueEncoders = nn.ModuleList(uniqueEncoders) # List of 2 - MLP encoders for the unique component of each modality
        self.norm = lambda x: nn.functional.normalize(x, dim=-1)
        self.add_shared = add_shared
        

    def forward(self, x1, x2):
        """
        Forward pass through the original JointOpt model that uses one decoder per disentangled component.
        Args:
        x1: Input data for modality 1.
        x2: Input data for modality 2.
        """
        
        s_12 = self.sharedEncoders[0](x1)  # Shared component from modality 1
        s_21 = self.sharedEncoders[1](x2)  # Shared component from modality 2

        # add probabilistic heads for shared components
        p_s_12_given_x1, mu1 = self.prob_heads[0](s_12)
        p_s_21_given_x2, mu2 = self.prob_heads[1](s_21)

        s_12_prob= p_s_12_given_x1.rsample()
        s_21_prob= p_s_21_given_x2.rsample()

        s_concat = torch.cat([s_12_prob.unsqueeze(dim=1), s_21_prob.unsqueeze(dim=1)], dim=1)

        if not self.add_shared:
            u_12 = self.uniqueEncoders[0](x1)  # Unique component from modality 1
            u_21 = self.uniqueEncoders[1](x2)  # Unique component from modality 2
        else:
            u_12 = self.uniqueEncoders[0](torch.cat([x1, s_12], dim= 1))  # Unique component from modality 1
            u_21 = self.uniqueEncoders[1](torch.cat([x2, s_21], dim= 1))  # Unique component from modality 2

        z_1_concat = torch.cat([u_12, s_21], dim=1)
        z_2_concat = torch.cat([u_21, s_12], dim=1)

        z_1_concat = self.norm(z_1_concat)
        z_2_concat = self.norm(z_2_concat)

        out = {"Z1": (u_12, s_21, s_12_prob), "Z2": (u_21, s_12, s_21_prob), \
                "s_concat": s_concat, "z_1_concat": z_1_concat, "z_2_concat": z_2_concat}
        return out