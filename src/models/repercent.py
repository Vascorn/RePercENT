import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
from typing import Literal, List
from src.DisentangledSSL.models import ProbabilisticEncoder 


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

        self.head12 = ProbabilisticEncoder(latent_dim, latent_dim, initialization='normal', distribution='vmf', vmfkappa=1e3)
        # self.decoder = decoder_model  # Decoder model to reconstruct original modalities
        # self.maping = {'z11': 0, 'z12': 1} # Mapping of the disentangled latent representations to the corresponding position in the perceiver output

    def forward(self, x):
        Z = self.encoder(x)  # Encode input data Xi to get latent representation Zi
        out = self.perceiver(Z)  # Pass latent representation through Perceiver to extract the disentangled features
        return out

class RePercENT(nn.Module):
    def __init__(self, M: int = 2, disenEncoder: List[DisenEncoder] = None, disen_mapping: dict[int, dict] = {'M1': {'z11': 0, 'z12': 1}, 'M2': {'z22': 0, 'z21': 1}}) -> None:
        super(RePercENT, self).__init__()

        assert M == len(disenEncoder), "Number of modalities M must match the length of disenEncoder list"

        self.M = M  # Number of modalities
        self.disenEncoders = nn.ModuleList(disenEncoder)  # List of DisenEncoder instances for each modality
        self.probab_heads = nn.ModuleList([]) # List of probabilistic heads for each shared disentangled factor 
        self.disen_mapping = disen_mapping
        self.iterations = 0

    def forward(self, x1, x2, v1, v2):
        """
        Forward pass through the RePercENT model.
        Args:
            x1: Input data for modality 1
            x2: Input data for modality 2
            v1: Augmented input data for modality 1
            v2: Augmented input data for modality 2
        """
        self.iterations += 1

        # The following assumes two modalities for simplicity - extension to M > 2 to be done
        Z1 = self.disenEncoders[0](x1)  # Encode modality 1
        Z2 = self.disenEncoders[1](x2)  # Encode modality 2

        # encode the augmented views
        Z1_v = self.disenEncoders[0](v1)  # Encode modality 1 augmented
        Z2_v = self.disenEncoders[1](v2)  # Encode modality 2 augmented


        # CONTINUE FROM HERE!!!!!!!!!!

        e1 = self.encoder_x1s(x1)
        e2 = self.encoder_x2s(x2)
        e1_v = self.encoder_x1s(v1)
        e2_v = self.encoder_x2s(v2)

        p_zs1_given_x1, mu1 = self.phead1(e1)
        p_zs2_given_x2, mu2 = self.phead2(e2)
        p_zsv1_given_v1, mu1_v = self.phead1(e1_v)
        p_zsv2_given_v2, mu2_v = self.phead2(e2_v)

        zs1 = p_zs1_given_x1.rsample()
        zs2 = p_zs2_given_x2.rsample()
        zsv1 = p_zsv1_given_v1.rsample()
        zsv2 = p_zsv2_given_v2.rsample()

        concat_embed = torch.cat([zs1.unsqueeze(dim=1), zs2.unsqueeze(dim=1)], dim=1)
        concat_embed_v = torch.cat([zsv1.unsqueeze(dim=1), zsv2.unsqueeze(dim=1)], dim=1)
        joint_loss, loss_x, loss_y = self.critic(concat_embed)
        joint_loss_v, loss_x_v, loss_y_v = self.critic(concat_embed_v)
        joint_loss = 0.5 * (joint_loss + joint_loss_v)
        loss_x = 0.5 * (loss_x + loss_x_v)
        loss_y = 0.5 * (loss_y + loss_y_v)
        loss_shared = joint_loss

        if self.condzs:
            z1x1 = self.encoder_x1(torch.cat([x1, e1], dim=1))
            z1xv1 = self.encoder_x1(torch.cat([v1, e1_v], dim=1))
            z2x2 = self.encoder_x2(torch.cat([x2, e2], dim=1))
            z2xv2 = self.encoder_x2(torch.cat([v2, e2_v], dim=1))
        else:
            z1x1 = self.encoder_x1(x1)
            z1xv1 = self.encoder_x1(v1)
            z2x2 = self.encoder_x2(x2)
            z2xv2 = self.encoder_x2(v2)

        if self.apdzs:
            if self.usezsx:
                zjointx1 = torch.cat([z1x1, e1], dim=1)
                zjointx2 = torch.cat([z2x2, e2], dim=1)
                zjointxv1 = torch.cat([z1xv1, e1_v], dim=1)
                zjointxv2 = torch.cat([z2xv2, e2_v], dim=1)
            else:
                zjointx1 = torch.cat([z1x1, e2], dim=1)
                zjointx2 = torch.cat([z2x2, e1], dim=1)
                zjointxv1 = torch.cat([z1xv1, e2_v], dim=1)
                zjointxv2 = torch.cat([z2xv2, e1_v], dim=1)

            if self.proj:
                zjointx1 = self.projection_x1(zjointx1)
                zjointx2 = self.projection_x2(zjointx2)
                zjointxv1 = self.projection_x1(zjointxv1)
                zjointxv2 = self.projection_x2(zjointxv2)

            zjointx1, zjointx2 = nn.functional.normalize(zjointx1, dim=-1), nn.functional.normalize(zjointx2, dim=-1)
            zjointxv1, zjointxv2 = nn.functional.normalize(zjointxv1, dim=-1), nn.functional.normalize(zjointxv2, dim=-1)
            concat_embed_x1 = torch.cat([zjointx1.unsqueeze(dim=1), zjointxv1.unsqueeze(dim=1)], dim=1)
            concat_embed_x2 = torch.cat([zjointx2.unsqueeze(dim=1), zjointxv2.unsqueeze(dim=1)], dim=1)
        else:
            z1x1_norm, z2x2_norm = nn.functional.normalize(z1x1, dim=-1), nn.functional.normalize(z2x2, dim=-1)
            z1xv1_norm, z2xv2_norm = nn.functional.normalize(z1xv1, dim=-1), nn.functional.normalize(z2xv2, dim=-1)
            concat_embed_x1 = torch.cat([z1x1_norm.unsqueeze(dim=1), z1xv1_norm.unsqueeze(dim=1)], dim=1)
            concat_embed_x2 = torch.cat([z2x2_norm.unsqueeze(dim=1), z2xv2_norm.unsqueeze(dim=1)], dim=1)

        specific_loss_x1, loss_x1, loss_y1 = self.critic(concat_embed_x1)
        specific_loss_x2, loss_x2, loss_y2 = self.critic(concat_embed_x2)

        loss_specific = specific_loss_x1 + specific_loss_x2

        if self.lmd_end_value > 0:
            lmd = self.lmd_scheduler(self.iterations)
        else:
            lmd = self.lmd_start_value

        loss_ortho = 0.5 * (ortho_loss(z1x1, e1, norm=self.ortho_norm) + ortho_loss(z2x2, e2, norm=self.ortho_norm)) + \
                    0.5 * (ortho_loss(z1xv1, e1_v, norm=self.ortho_norm) + ortho_loss(z2xv2, e2_v, norm=self.ortho_norm))
        
        loss = 2 * loss_shared/(1+self.a) + self.a * loss_specific/(1+self.a) + lmd * loss_ortho

        return loss, {'loss': loss.item(), 'shared': loss_shared.item(), 'clip': joint_loss.item(), 'loss_x': loss_x.item(), 'loss_y': loss_y.item(),
                       'specific': loss_specific.item(), 'ortho': loss_ortho.item(), 'lmd': lmd}