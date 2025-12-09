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


class simpleEncoder(nn.Module):
    def __init__(self, input_dim: int = 64, latent_dim: int = 32, activation: typing.Literal['relu', 'tanh', 'sigmoid'] = 'relu'):
        super(simpleEncoder, self).__init__()
        self.input_dim = input_dim # initial input dimension
        self.hidden_dim = latent_dim # hidden layer dimension
        self.latent_dim = latent_dim # final output dimension

        match activation:
            case 'relu':
                self.activation = nn.ReLU()
            case 'gelu':
                self.activation = nn.GELU()
            case 'sigmoid':
                self.activation = nn.Sigmoid()
            case _:
                exit ("Unsupported activation function")

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, self.latent_dim),
            self.activation
        )

    def forward(self, x):
        return self.mlp(x)

# Initial version for RePercENT model that handles two modalities
class DisenEncoder(nn.Module):
    def __init__(self, encoder_model= None, perceiver_model=None):
        super().__init__()
        self.encoder = encoder_model
        self.perceiver = perceiver_model


    def forward(self, x):
        Z = self.encoder(x)  # Encode input data Xi to get latent representation Z
        out = self.perceiver(Z)  # Pass latent representation through Perceiver to extract the disentangled features
        return out

    def __repr__(self):
        print(f"DisenEncoder with encoder: {self.encoder} and perceiver: {self.perceiver}")
        return

class RePercENT(nn.Module):
    def __init__(self, M: int = 2, disenEncoder: List[DisenEncoder] = None, \
        disen_mapping: dict[int, dict] = {'M_1': {'U_12': 0, 'S_12': 1}, 'M_2': {'U_21': 0, 'S_21': 1}}) -> None:
        '''
        RePercENT model for multi-modal representation learning with disentangled factors.
        Args:
            M (int): Number of modalities. Default is 2.
            disenEncoder (List[DisenEncoder]): List of DisenEncoder instances for each modality.
            disen_mapping (dict): Mapping the position of each disentangled factor to the corresponding position in the output of each Perceiver based encoder (disenEncoder). 
                                The keys are the different modalities (e.g., 'M_1', 'M_2', ..., 'M_N'), and the values are dictionaries that map the names of the disentangled 
                                factors (e.g., 'U_12' -> unique component of modality 1 with respect to modality 2, 'S_12' -> shared component between modality 1 and 2, etc.) 
                                to their respective indices in the output tensor of the corresponding DisenEncoder.
        '''

        super().__init__()

        assert M == len(disenEncoder), "Number of modalities M must match the length of disenEncoder list"
        
        self.M = M  # Number of modalities
        # self.latent_dim = disenEncoder[0].perceiver.latents.shape[-1]  # All DisenEncoders must have the same latent dimension

        # for de in disenEncoder:
        #     assert de.perceiver.latents.shape[-1] == self.latent_dim, "All DisenEncoders must have the same latent dimension"

        self.disenEncoders = nn.ModuleList(disenEncoder)  # List of DisenEncoder instances for each modality
        self.prob_heads = nn.ModuleList([ProbabilisticEncoder(nn.Identity(), distribution= "vmf", vmfkappa= 1e3) for _ in range(self.M)])  # Probabilistic heads for each of S_12 and S_21 - assuming only two modalities

        self.disen_mapping = disen_mapping
        self.norm = lambda x: nn.functional.normalize(x, dim=-1)
        

    def forward(self, x1, x2):
        """
        Forward pass through the RePercENT model.
        Args:
        x1: Input data for modality 1.
        x2: Input data for modality 2.
        """
        
        Z1 = self.disenEncoders[0](x1)  # Encode modality 1
        Z2 = self.disenEncoders[1](x2)  # Encode modality 2

        
        u_12_pos = self.disen_mapping['M_1']['U_12']
        s_12_pos = self.disen_mapping['M_1']['S_12']
        u_21_pos = self.disen_mapping['M_2']['U_21']
        s_21_pos = self.disen_mapping['M_2']['S_21']

        # extract each component
        #unique
        u_12 = Z1[:, u_12_pos, :]  # Unique component from modality 1
        u_21 = Z2[:, u_21_pos, :]  # Unique component from modality 2
        # shared
        s_12 = Z1[:, s_12_pos, :]  # Shared component from modality 1
        s_21 = Z2[:, s_21_pos, :]  # Shared component from modality 2

        # add probabilistic heads for shared components
        p_s_12_given_x1, mu1 = self.prob_heads[0](s_12)
        p_s_21_given_x2, mu2 = self.prob_heads[1](s_21)

        s_12_prob= p_s_12_given_x1.rsample()
        s_21_prob= p_s_21_given_x2.rsample()

        s_concat = torch.cat([s_12_prob.unsqueeze(dim=1), s_21_prob.unsqueeze(dim=1)], dim=1) # THIS IS THE ORIGINAL
        # s_concat = torch.cat([s_12.unsqueeze(dim=1), s_21.unsqueeze(dim=1)], dim=1) # MODIFIED THAT USES NO PROBABILISTIC SAMPLING
        # s_concat = self.norm(s_concat)
        # now compute the losses for the specific components

        z_1_concat = torch.cat([u_12, s_21], dim=1) # THIS IS THE ORIGINAL
        z_2_concat = torch.cat([u_21, s_12], dim=1)
        # z_1_concat = torch.cat([u_12, s_12], dim=1) # MODIFIED TO USE THE IDENTICAL SHARED COMPONENTS
        # z_2_concat = torch.cat([u_21, s_21], dim=1)

        # normalize the joint representations across last dimension
        z_1_concat = self.norm(z_1_concat)
        z_2_concat = self.norm(z_2_concat)

        out = {"Z1": (u_12, s_21, s_12_prob), "Z2": (u_21, s_12, s_21_prob), \
                "s_concat": s_concat, "z_1_concat": z_1_concat, "z_2_concat": z_2_concat}
        return out


# This function with calculate the custom pairwise loss for the RePercENT model
# It is based on the JointDisenModel from the DisentangledSSL package (https://github.com/uhlerlab/DisentangledSSL)
class DisenLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, lmd: float= 0.5, lmd_start_value: float= 1e-3, lmd_end_value: float= 1, lmd_n_iterations: int=1e4, lmd_start_iteration: int=5e3, ortho_norm: bool= True) -> None:
        super().__init__()
        self.critic = SupConLoss()
        self.norm = lambda x: nn.functional.normalize(x, dim=-1)
        self.alpha = alpha
        self.lmd = lmd
        self.lmd_scheduler = None if lmd_end_value <= 0 else ExponentialScheduler(start_value=lmd_start_value, end_value=lmd_end_value,
                                                             n_iterations=lmd_n_iterations, start_iteration=lmd_start_iteration)

        self.lmd_start_value = lmd_start_value
        self.lmd_end_value = lmd_end_value
        self.iterations = 0
        self.ortho_norm = ortho_norm
        self.ortho_loss = lambda x, y: torch.norm(torch.matmul(self.norm(x).T, self.norm(y))) if self.ortho_norm else NotImplementedError('Please set norm=True')

    def forward(self, outputs, outputs_aug):
        """
        Compute the disentanglement loss for the RePercENT model.
        Args:
            outputs: Output dictionary from the forward pass of the RePercENT model.
            outputs_aug: Output dictionary from the forward pass of the RePercENT model with augmented data.
        Returns:
            loss: Total loss as a scalar tensor.
            
        """
        self.iterations += 1
        u_12, s_21, s_12_prob = outputs["Z1"]
        u_21, s_12, s_21_prob = outputs["Z2"]
        s_concat = outputs["s_concat"]
        z_1_concat = outputs["z_1_concat"]
        z_2_concat = outputs["z_2_concat"]

        u_12_aug, s_21_aug, s_12_prob_aug = outputs_aug["Z1"]
        u_21_aug, s_12_aug, s_21_prob_aug = outputs_aug["Z2"]
        s_concat_aug = outputs_aug["s_concat"]
        z_1_concat_aug = outputs_aug["z_1_concat"]
        z_2_concat_aug = outputs_aug["z_2_concat"]

        # Calculate shared component losses
        shared_loss, loss_x, loss_y = self.critic(s_concat)
        shared_loss_aug, loss_x_aug, loss_y_aug = self.critic(s_concat_aug)
        joint_loss = (shared_loss + shared_loss_aug) / 2
        loss_x = (loss_x + loss_x_aug) / 2
        loss_y = (loss_y + loss_y_aug) / 2

        # Calculate losses for modality unique components
        concat_embed_x1 = torch.cat([z_1_concat.unsqueeze(dim= 1), z_1_concat_aug.unsqueeze(dim= 1)], dim= 1)
        concat_embed_x2 = torch.cat([z_2_concat.unsqueeze(dim= 1), z_2_concat_aug.unsqueeze(dim= 1)], dim= 1)

        unique_loss_x1, loss_x1, loss_y1 = self.critic(concat_embed_x1)
        unique_loss_x2, loss_x2, loss_y2 = self.critic(concat_embed_x2)
        unique_loss = (unique_loss_x1 + unique_loss_x2) / 2

        # Calculate orthogonality loss
        loss_ortho = 0.5 * (self.ortho_loss(u_12, s_12) + self.ortho_loss(u_21, s_21)) + \
                     0.5 * (self.ortho_loss(u_12_aug, s_12_aug) + self.ortho_loss(u_21_aug, s_21_aug))

        # Total loss
        if self.lmd_scheduler is not None:
            self.lmd = self.lmd_scheduler(self.iterations)

        loss = 2 * joint_loss / (1 + self.alpha) + self.alpha * unique_loss / (1 + self.alpha) + self.lmd * loss_ortho

        loss_logs = {'loss': loss.item(),
                     'shared': joint_loss.item(),
                     'loss_x': loss_x.item(),
                     'loss_y': loss_y.item(),
                     'unique': unique_loss.item(),
                     'ortho': loss_ortho.item(),
                     'lmd': self.lmd
                     }
        return loss, loss_logs

        