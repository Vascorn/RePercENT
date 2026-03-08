import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
import typing
from typing import Literal, List
from src.DisentangledSSL.models import ProbabilisticEncoder 
from src.DisentangledSSL.losses import SupConLoss, ortho_loss, kl_vmf
from src.DisentangledSSL.utils import ExponentialScheduler
from itertools import permutations
from src.models.jointopt_2m import MLP

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
        out = self.mlp(x)
        out = nn.functional.reshape(out, (-1, 1, self.latent_dim))
        return out

# Initial version for RePercENT model that handles two modalities
class DisenEncoder(nn.Module):
    def __init__(self, encoder_model= None, perceiver_model=None):
        super().__init__()
        self.encoder = encoder_model
        self.perceiver = perceiver_model


    def forward(self, x, mask= None, pos_enc= None):
        Z = self.encoder(x)  # Encode input data Xi to get latent representation Z
        out = self.perceiver(Z, mask= mask, pos_enc= pos_enc)  # Pass latent representation through Perceiver to extract the disentangled features
        return out

    
class RePercENT(nn.Module):
    def __init__(self, M: int = 2, disenEncoder: List[DisenEncoder] = None, \
        disenDecoder: List[nn.Module] = None, \
        disen_mapping: dict[int, dict] = {'M_1': {'U_12': 0, 'S_12': 1}, 'M_2': {'U_21': 0, 'S_21': 1}},
        vmfkappa: float= 1e3, add_pos_encoding: bool = True) -> None:
        '''
        RePercENT model for multi-modal representation learning with disentangled factors.
        Args:
            M (int): Number of modalities. Default is 2.
            disenEncoder (List[DisenEncoder]): List of DisenEncoder instances for each modality.
            disenDecoder (List[nn.Module]): List of decoder modules for each modality if recon is True.
            disen_mapping (dict): Mapping the position of each disentangled factor to the corresponding position in the output of each Perceiver based encoder (disenEncoder). 
                                The keys are the different modalities (e.g., 'M_1', 'M_2', ..., 'M_N'), and the values are dictionaries that map the names of the disentangled 
                                factors (e.g., 'U_12' -> unique component of modality 1 with respect to modality 2, 'S_12' -> shared component between modality 1 and 2, etc.) 
                                to their respective indices in the output tensor of the corresponding DisenEncoder.
            vmfkappa (float): Concentration parameter for the vMF distribution in the probabilistic encoder heads. Default is 1e3.
            add_pos_encoding (bool): Whether to add learnable positional encodings to the outputs of the Perceiver encoders before extracting the disentangled factors. Default is True.
        '''

        super().__init__()

        assert M == len(disenEncoder), "Number of modalities M must match the length of disenEncoder list"
        
        self.M = M  # Number of modalities

        # for de in disenEncoder:
        #     assert de.perceiver.latents.shape[-1] == self.latent_dim, "All DisenEncoders must have the same latent dimension"

        self.disenEncoders = nn.ModuleList(disenEncoder)  # List of DisenEncoder instances for each modality
        

        # self.prob_heads = nn.ModuleList([ProbabilisticEncoder(nn.Identity(), distribution= "vmf", vmfkappa= 1e3) for _ in range(self.M)])  # Probabilistic heads for each of S_12 and S_21 - assuming only two modalities
        
        self.latent_dim = disenEncoder[0].perceiver.latents.shape[-1]  # Latent dimension of the representations
        self.seq_dim = disenEncoder[0].perceiver.seq_dim # Sequence dimension of the representations
        print(f"RePercENT model initialized with latent dimension: {self.latent_dim}")
        # create probalitistic heads for each shared component S_ij

        self.prob_heads = nn.ModuleList()
        # save the order of (i,j) pairs for the probabilistic heads
        perm = torch.tensor(list(permutations(range(self.M), 2)), dtype=torch.long)  # 0-based
        self.register_buffer("perm_i", perm[:, 0], persistent=False)
        self.register_buffer("perm_j", perm[:, 1], persistent=False)

        for i, j in zip(self.perm_i, self.perm_j):
            self.prob_heads.append(ProbabilisticEncoder(nn.Identity(), distribution= "vmf", vmfkappa= vmfkappa))
                                                        
            
        self.disen_mapping = disen_mapping

        self.norm = lambda x: nn.functional.normalize(x, dim=-1)

        
        # ========== Positional Encodings ==========
        self.add_pos_encoding = add_pos_encoding
        if self.add_pos_encoding:
            # Build mapping from ordered pairs (i,j) to unique indices (1-based modality indices)
            pair_to_idx = {}
            pair_idx = 0
            for i in range(1, M + 1):
                for j in range(1, M + 1):
                    if i != j:
                        pair_to_idx[(i, j)] = pair_idx
                        pair_idx += 1
            num_pairs = len(pair_to_idx)  # M * (M - 1)

            # Learnable positional encodings
            self.pair_pos_enc = nn.Parameter(torch.randn(num_pairs, self.latent_dim) * 0.02)
            self.type_pos_enc = nn.Parameter(torch.randn(2, self.latent_dim) * 0.02)  # 0: U (unique), 1: S (shared)


            # indices for all unordered pairs i<j (0-based)
            idx = torch.triu_indices(self.M, self.M, offset=1)  # (2, P)
            self.register_buffer("pair_i", idx[0])  # (P,)
            self.register_buffer("pair_j", idx[1])  # (P,)

        # Pre-compute index buffers for each modality following disen_mapping order
        for m in range(1, M + 1):
            mapping = disen_mapping[f"M_{m}"]
            seq_len = len(mapping)
            p_idx = torch.zeros(seq_len, dtype=torch.long)
            t_idx = torch.zeros(seq_len, dtype=torch.long)

            for comp_name, pos in mapping.items():
                comp_type = comp_name[0]  # 'U' or 'S'
                # Extract i and j from component name (e.g., "U_12" -> i=1, j=2)
                pair_str = comp_name.split('_')[1]  # "12"
                i_comp, j_comp = int(pair_str[0]), int(pair_str[1])

                p_idx[pos] = pair_to_idx[(i_comp, j_comp)]
                t_idx[pos] = 0 if comp_type == 'U' else 1

            self.register_buffer(f"pair_idx_m{m}", p_idx, persistent=False)
            self.register_buffer(f"type_idx_m{m}", t_idx, persistent=False)
        self.P = idx.shape[1]

    def get_slot(self, Zi, i: int, comp: str):
        """Zi: output of the disenEncoder for modality modality i, representing component <comp>."""
        pos = self.disen_mapping[f"M_{i}"][comp]
        return Zi[:, pos, :] 

    def forward(self, x, mask= None):
        """
        Forward pass through the RePercENT model.
        Args:
            x: List of input data for each modality. Length of the list should be equal to M.
            mask: Optional list of masks for cross-modal attention in the Perceiver encoders.
        """
        
        assert len(x) == self.M, "Input data length must match number of modalities M"
        
        # Compute positional encodings for each modality and pass to encoders
        Z = []
        for m in range(self.M):
            if self.add_pos_encoding:
                p_idx = getattr(self, f"pair_idx_m{m + 1}")  # (seq_dim,)
                t_idx = getattr(self, f"type_idx_m{m + 1}")  # (seq_dim,)
                pair_pe = self.pair_pos_enc[p_idx]  # (seq_dim, latent_dim)
                type_pe = self.type_pos_enc[t_idx]  # (seq_dim, latent_dim)
                pos_enc = pair_pe + type_pe  # (seq_dim, latent_dim)
            else:
                pos_enc = None
            Z.append(self.disenEncoders[m](x[m], mask=mask[m], pos_enc=pos_enc))
        
        # extract components:
        # Each of U[*, i, j, *] corresponds to unique component from modality i with respect to modality j, similar for S and S_prob
        U = torch.zeros((x[0].shape[0], self.M, self.M, Z[0].shape[-1]), device= Z[0].device)  # Initialize tensor to hold unique components
        S_view = torch.zeros((x[0].shape[0], self.M, self.M, Z[0].shape[-1]), device= Z[0].device)  # Initialize tensor to hold shared components
        S_prob = torch.zeros((x[0].shape[0], self.M, self.M, Z[0].shape[-1]), device= Z[0].device)  # Initialize tensor to hold probabilistic shared components


        for n, (i, j) in enumerate(zip(self.perm_i + 1, self.perm_j + 1)):  # convert to 1-based for mapping
            u_ij = self.get_slot(Z[i-1], i, f"U_{i}{j}")
            s_ij = self.get_slot(Z[i-1], i, f"S_{i}{j}")
            
            p_s_ij_given_xi, _ = self.prob_heads[n](s_ij)
            s_prob_ij = p_s_ij_given_xi.rsample()

            U[:, i-1, j-1, :] = u_ij
            S_view[:, i-1, j-1, :] = s_ij
            S_prob[:, i-1, j-1, :] = s_prob_ij


        # --- S_concat: (B, P, 2, D) = [s_ij, s_ji] ---
        i = self.pair_i
        j = self.pair_j
        S_concat = torch.stack([S_prob[:, i, j, :], S_prob[:, j, i, :]], dim=2)  # (B,P,2,D)
        S_concat = self.norm(S_concat)


        # --- Z_concat: (B, P, 2, 2D) ---
        # view 0 for pair (i,j): [u_ij, s_ji]
        Z_i_concat = torch.cat([U[:, i, j, :], S_view[:, j, i, :]], dim=-1)  # (B,P,2D)
        Z_i_concat = self.norm(Z_i_concat)
        # view 1 for pair (i,j): [u_ji, s_ij]
        Z_j_concat = torch.cat([U[:, j, i, :], S_view[:, i, j, :]], dim=-1)  # (B,P,2D)
        Z_j_concat = self.norm(Z_j_concat)

        
        out = {"U": U, "S_view": S_view, "S_prob": S_prob, "S_concat": S_concat, "Z_i_concat": Z_i_concat, "Z_j_concat": Z_j_concat}
            
        return out
    



# This function with calculate the custom pairwise loss for the RePercENT model
# It is based on the JointDisenModel from the DisentangledSSL package (https://github.com/uhlerlab/DisentangledSSL)
class DisenLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, 
                lmd: float= 0.5, lmd_start_value: float= 1e-3, 
                lmd_end_value: float= 1, lmd_n_iterations: int=1e4, 
                lmd_start_iteration: int=5e3, ortho_norm: bool= True, 
                M: int= 2) -> None:
        super().__init__()
        self.critic = SupConLoss()
        self.norm = lambda x: nn.functional.normalize(x, dim=-1)
        self.alpha = alpha
        self.lmd = lmd if lmd_end_value <= 0 else lmd_start_value
        self.lmd_scheduler = None if lmd_end_value <= 0 else ExponentialScheduler(start_value=lmd_start_value, end_value=lmd_end_value,
                                                             n_iterations=lmd_n_iterations, start_iteration=lmd_start_iteration)
        if self.lmd_scheduler is not None:
            print(f"Initialized lambda scheduler with start value {lmd_start_value}, end value {lmd_end_value}, n_iterations {lmd_n_iterations}, and start_iteration {lmd_start_iteration}")
        self.lmd_start_value = lmd_start_value
        self.lmd_end_value = lmd_end_value
        self.iterations = 0
        self.ortho_norm = ortho_norm
        self.M = M

        # indices for all unordered pairs i<j (0-based)
        self.idx = torch.triu_indices(self.M, self.M, offset=1)  # (2, P)
        self.register_buffer("pair_i", self.idx[0])  # (P,)
        self.register_buffer("pair_j", self.idx[1])  # (P,)
        self.P = self.idx.shape[1]

    def ortho_loss(self,x, y, norm= True, ltype: Literal["standard", "cosim", "xcov"] = "xcov", eps: float= 1e-8):
        if norm:
            x = self.norm(x)
            y = self.norm(y)

        B, D= x.shape

        match ltype:
            case "standard":
                res = torch.matmul(x.T, y)  # (D, D)
                res = torch.linalg.norm(res, ord="fro")
            case "cosim":                
                res = torch.matmul(x, y.T)  # (B, B)    
                res = torch.linalg.norm(res, ord="fro") / (B + eps)
            case "xcov":
                x_centered = x - x.mean(dim=0, keepdim=True)
                y_centered = y - y.mean(dim=0, keepdim=True)

                x_norm = x_centered / (x_centered.std(dim=0, keepdim=True) + eps)
                y_norm = y_centered / (y_centered.std(dim=0, keepdim=True) + eps)

                cov = torch.matmul(x_norm.T, y_norm) / (B + eps)  # (D, D)
                res = torch.linalg.norm(cov, ord="fro") / D
            case _:
                raise ValueError(f"Unsupported orthogonality loss type: {ltype}")
        
        return res
    

    def pairwise_loss(self, outputs, outputs_aug):
        """
        Compute the disentanglement loss for the RePercENT model.
        Args:
            outputs: Output dictionary from the forward pass of the RePercENT model.
            outputs_aug: Output dictionary from the forward pass of the RePercENT model with augmented data.
        Returns:
            loss: Total loss as a scalar tensor.
            
        """
        u_ij, s_ji, s_ij_prob = outputs["Zi"]
        u_ji, s_ij, s_ji_prob = outputs["Zj"]
        s_concat = outputs["s_concat"]
        z_i_concat = outputs["z_i_concat"]
        z_j_concat = outputs["z_j_concat"]
        u_ij_aug, s_ji_aug, s_ij_prob_aug = outputs_aug["Zi"]
        u_ji_aug, s_ij_aug, s_ji_prob_aug = outputs_aug["Zj"]
        s_concat_aug = outputs_aug["s_concat"]
        z_i_concat_aug = outputs_aug["z_i_concat"]
        z_j_concat_aug = outputs_aug["z_j_concat"]

        # Calculate shared component losses
        shared_loss, loss_x, loss_y = self.critic(s_concat)
        shared_loss_aug, loss_x_aug, loss_y_aug = self.critic(s_concat_aug)
        joint_loss = (shared_loss + shared_loss_aug) / 2
        loss_x = (loss_x + loss_x_aug) / 2
        loss_y = (loss_y + loss_y_aug) / 2

        #Calculate kl divergence for the probabilistic shared components
        #skl = kl_vmf(self.norm(s_ij), self.norm(s_ji))
        #skl_aug = kl_vmf(self.norm(s_ij_aug), self.norm(s_ji_aug))
        #skl = (skl + skl_aug) / 2

        # Calculate losses for modality unique components
        concat_embed_xi = torch.cat([z_i_concat.unsqueeze(dim= 1), z_i_concat_aug.unsqueeze(dim= 1)], dim= 1)
        concat_embed_xj = torch.cat([z_j_concat.unsqueeze(dim= 1), z_j_concat_aug.unsqueeze(dim= 1)], dim= 1)

        unique_loss_xi, loss_xi, loss_yi = self.critic(concat_embed_xi)
        unique_loss_xj, loss_xj, loss_yj = self.critic(concat_embed_xj)
        unique_loss = (unique_loss_xi + unique_loss_xj) / 2


        # Calculate orthogonality loss
        loss_ortho = 0.5 * (self.ortho_loss(u_ij, s_ij) + self.ortho_loss(u_ji, s_ji)) + \
                     0.5 * (self.ortho_loss(u_ij_aug, s_ij_aug) + self.ortho_loss(u_ji_aug, s_ji_aug))
        # Total loss
        if self.lmd_scheduler is not None:
            self.lmd = self.lmd_scheduler(self.iterations)

        loss = 2 * joint_loss / (1 + self.alpha) + self.alpha * unique_loss / (1 + self.alpha) + self.lmd * loss_ortho #+ 0.1 * skl

        # We also log a fixed weight version of the loss for easier comparison across training when alpha, lambda are using schedulers
        fixed_weight_loss = joint_loss + unique_loss + loss_ortho

        loss_logs = {'loss': loss.item(),
                     'shared': joint_loss.item(),
                     'loss_x': loss_x.item(),
                     'loss_y': loss_y.item(),
                     'unique': unique_loss.item(),
                     'ortho': loss_ortho.item(),
                     'fw_loss': fixed_weight_loss.item(),
                     'lmd': self.lmd
                     }

        return loss, loss_logs


    def forward(self, outputs, outputs_aug):
        # one scheduler step per batch
        self.iterations += 1
        if self.lmd_scheduler is not None:
            self.lmd = self.lmd_scheduler(self.iterations)

        total_loss = 0.0
        total_logs = {}

        # loop over unordered pairs via p index
        for p in range(self.P):
            i = int(self.pair_i[p].item())  # 0-based
            j = int(self.pair_j[p].item())  # 0-based
            
            # build per-pair views (original)
            Zi = (outputs["U"][:, i, j, :], outputs["S_view"][:, j, i, :], outputs["S_prob"][:, j, i, :])
            Zj = (outputs["U"][:, j, i, :], outputs["S_view"][:, i, j, :], outputs["S_prob"][:, i, j, :])
            outputs_pair = {
                "Zi": Zi,
                "Zj": Zj,
                "s_concat": outputs["S_concat"][:, p, :, :],       # (B,2,D)
                "z_i_concat": outputs["Z_i_concat"][:, p, :],      # (B,2D)
                "z_j_concat": outputs["Z_j_concat"][:, p, :]      # (B,2D)
            }

            # build per-pair views (aug)
            Zi = (outputs_aug["U"][:, i, j, :], outputs_aug["S_view"][:, j, i, :], outputs_aug["S_prob"][:, j, i, :])
            Zj = (outputs_aug["U"][:, j, i, :], outputs_aug["S_view"][:, i, j, :], outputs_aug["S_prob"][:, i, j, :])
            outputs_pair_aug = {
                "Zi": Zi,
                "Zj": Zj,
                "s_concat": outputs_aug["S_concat"][:, p, :, :],
                "z_i_concat": outputs_aug["Z_i_concat"][:, p, :],
                "z_j_concat": outputs_aug["Z_j_concat"][:, p, :]
            }

            l, logs = self.pairwise_loss(outputs_pair, outputs_pair_aug)
            total_loss = total_loss + l

            for k, v in logs.items():
                total_logs[k] = total_logs.get(k, 0.0) + v

        # average over pairs (usually what you want)
        total_loss = total_loss / self.P

        for k in total_logs:
            total_logs[k] /= self.P

        return total_loss, total_logs


        