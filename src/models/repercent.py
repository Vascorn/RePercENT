import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
from typing import Literal, List
from src.DisentangledSSL.models import ProbabilisticEncoder 
from src.DisentangledSSL.losses import SupConLoss, ortho_loss


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
    def __init__(self, input_dim, latent_dim, num_latents= 2, encoder_model= None, perceiver_model=None):
        super(DisenEncoder, self).__init__()
        self.encoder = encoder_model
        self.perceiver = perceiver_model
        self.t = self.perceiver.input_channels # the temporal dimension of the i^th modality (Z \in R^{t x d})
        self.d = self.perceiver.input_channels  # the feature dimension of the i^th modality (Z \in R^{t x d})


    def forward(self, x):
        Z = self.encoder(x)  # Encode input data Xi to get latent representation Zi
        out = self.perceiver(Z)  # Pass latent representation through Perceiver to extract the disentangled features
        return out

class RePercENT(nn.Module):
    def __init__(self, M: int = 2, disenEncoder: List[DisenEncoder] = None, \
        disen_mapping: dict[int, dict] = {'M_1': {'U_12': 0, 'S_12': 1}, 'M_2': {'U_21': 0, 'S_21': 1}},
        a: float = 1.0, lmd: float= 0.5, ortho_norm: bool= True) -> None:
        '''
        RePercENT model for multi-modal representation learning with disentangled factors.
        Args:
            M (int): Number of modalities.
            disenEncoder (List[DisenEncoder]): List of DisenEncoder instances for each modality.
            disen_mapping (dict): Mapping the position of each disentangled factor to the corresponding position in the output of each Perceiver based encoder (disenEncoder). 
                                The keys are the different modalities (e.g., 'M_1', 'M_2', ..., 'M_N'), and the values are dictionaries that map the names of the disentangled 
                                factors (e.g., 'U_12' -> unique component of modality 1 with respect to modality 2, 'S_12' -> shared component between modality 1 and 2, etc.) 
                                to their respective indices in the output tensor of the corresponding DisenEncoder.
        '''

        super(RePercENT, self).__init__()

        assert M == len(disenEncoder), "Number of modalities M must match the length of disenEncoder list"
        
        self.M = M  # Number of modalities
        self.latent_dim = disenEncoder[0].perceiver.latent_dim  # All DisenEncoders must have the same latent dimension

        for de in disenEncoder:
            assert de.perceiver.latent_dim == self.latent_dim, "All DisenEncoders must have the same latent dimension"

        self.disenEncoders = nn.ModuleList(disenEncoder)  # List of DisenEncoder instances for each modality
        self.prob_heads = nn.ModuleList([ProbabilisticEncoder(nn.Identity(), distribution= "vmf") for _ in range(self.M)])  # Probabilistic heads for each of S_12 and S_21 - assuming only two modalities

        self.disen_mapping = disen_mapping
        self.iterations = 0
        self.critic = SupConLoss()
        self.norm = lambda x: nn.functional.normalize(x, dim=-1)
        self.a = alpha
        self.lmd = lmd
        self.ortho_norm = ortho_norm
        self.loss_ortho = lambda x, y: ortho_loss(x, y, norm= self.ortho_norm)

    def forward(self, x1, x2, v1, v2):
        """
        Forward pass through the RePercENT model.
        Args:
        x1: Input data for modality 1.
        x2: Input data for modality 2.
        v1: Augmented view of input data for modality 1.
        v2: Augmented view of input data for modality 2.
        """
        self.iterations += 1
        
        Z1 = self.disenEncoders[0](x1)  # Encode modality 1
        Z2 = self.disenEncoders[1](x2)  # Encode modality 2

        # encode the augmented views
        Z1_v = self.disenEncoders[0](v1)  # Encode modality 1 augmented
        Z2_v = self.disenEncoders[1](v2)  # Encode modality 2 augmented

        u_12_pos = self.disen_mapping['M_1']['U_12']
        s_12_pos = self.disen_mapping['M_1']['S_12']
        u_21_pos = self.disen_mapping['M_2']['U_21']
        s_21_pos = self.disen_mapping['M_2']['S_21']

        # extract each component
        #unique
        u_12 = Z1[u_12_pos, :]  # Unique component from modality 1
        u_12_v = Z1_v[u_12_pos, :]  # Unique component from modality 1 augmented
        u_21 = Z2[u_21_pos, :]  # Unique component from modality 2
        u_21_v = Z2_v[u_21_pos, :]  # Unique component from modality 2 augmented
        # shared
        s_12 = Z1[s_12_pos, :]  # Shared component from modality 1
        s_12_v = Z1_v[s_12_pos, :]  # Shared component from modality 1 augmented
        s_21 = Z2[s_21_pos, :]  # Shared component from modality 2
        s_21_v = Z2_v[s_21_pos, :]  # Shared component from modality 2 augmented

        # add probabilistic heads for shared components
        p_s_12_given_x1, mu1 = self.prob_heads[0](s_12)
        p_s_21_given_x2, mu2 = self.prob_heads[1](s_21)
        p_s_12_v_given_v1, mu1_v = self.prob_heads[0](s_12_v)
        p_s_21_v_given_v2, mu2_v = self.prob_heads[1](s_21_v)

        s_12 = p_s_12_given_x1.rsample()
        s_21 = p_s_21_given_x2.rsample()
        s_12_v = p_s_12_v_given_v1.rsample()
        s_21_v = p_s_21_v_given_v2.rsample()

        concat_embed = torch.cat([s_12.unsqueeze(dim=1), s_21.unsqueeze(dim=1)], dim=1)
        concat_embed_v = torch.cat([s_12_v.unsqueeze(dim=1), s_21_v.unsqueeze(dim=1)], dim=1)
        joint_loss, loss_x, loss_y = self.critic(concat_embed)
        joint_loss_v, loss_x_v, loss_y_v = self.critic(concat_embed_v)
        joint_loss = 0.5 * (joint_loss + joint_loss_v)
        loss_x = 0.5 * (loss_x + loss_x_v)
        loss_y = 0.5 * (loss_y + loss_y_v)
        loss_shared = joint_loss


        # now compute the losses for the specific components

        zjoint1 = torch.cat([u_12, Z2[s_21_pos, :]], dim=1)
        zjoint2 = torch.cat([u_21, Z1[s_12_pos, :]], dim=1)
        zjoint1_v = torch.cat([u_12_v, Z2_v[s_21_pos, :]], dim=1)
        zjoint2_v = torch.cat([u_21_v, Z1_v[s_12_pos, :]], dim=1)

        # normalize the joint representations across last dimension
        zjoint1 = self.norm(zjoint1)
        zjoint2 = self.norm(zjoint2)
        zjoint1_v = self.norm(zjoint1_v)
        zjoint2_v = self.norm(zjoint2_v)

        concat_embed_x1 = torch.cat([zjoint1.unsqueeze(dim=1), zjoint2.unsqueeze(dim=1)], dim=1)
        concat_embed_x2 = torch.cat([zjoint1_v.unsqueeze(dim=1), zjoint2_v.unsqueeze(dim=1)], dim=1)
        

        unique_loss_x1, loss_x1, loss_y1 = self.critic(concat_embed_x1)
        unique_loss_x2, loss_x2, loss_y2 = self.critic(concat_embed_x2)
        unique_loss = unique_loss_x1 + unique_loss_x2

        loss_ortho = 0.5 * (self.ortho_loss(u_12, Z1[s_12_pos, :]) + self.ortho_loss(u_21, Z2[s_21_pos, :])) + \
            0.5 * (self.ortho_loss(u_12_v, Z1_v[s_12_pos, :]) + self.ortho_loss(u_21_v, Z2_v[s_21_pos, :]))

        loss = 2 * loss_shared/(1+ self.a) + self.a * unique_loss/(1+ self.a) + self.lmd * loss_ortho


        # update logs
        loss_logs = {'loss': loss.item(),
                     'shared': loss_shared.item(),
                     'clip': joint_loss.item(),
                     'loss_x': loss_x.item(),
                     'loss_y': loss_y.item(),
                     'unique': unique_loss.item(),
                     'ortho': loss_ortho.item()
                     }

        return loss, loss_logs