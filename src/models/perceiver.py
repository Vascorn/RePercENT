from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    
    x = x.unsqueeze(-1)
    
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype) # change this line. no need to create every time
    
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    
    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    # Implementation of Gated Linear Unit with GELU (Generalized Linear Unit) activation
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 2, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MoEFeedForward(nn.Module):
    """
    Token-level MoE FFN for x: (B, N, D)
    - Routes each token (latent) to top-k experts
    - Combines expert outputs with gate weights
    """
    def __init__(
        self,
        dim: int,
        mult: int = 2,
        num_experts: int = 8,
        temperature: float = 1.0, 
        hard: bool = False, 
        top_k: int = 2,
        dropout: float = 0.0,
        gate_dropout: float = 0.0,
        use_softmax_gating: bool = True,
    ):
        super().__init__()
        assert top_k <= num_experts and top_k >= 1
        self.dim = dim
        self.num_experts = num_experts
        self.temperature = temperature
        self.hard = hard
        self.top_k = top_k
        self.use_softmax_gating = use_softmax_gating

        # router: produces logits over experts for each token
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.gate_dropout = nn.Dropout(gate_dropout)

        # experts: same FFN architecture you used (GEGLU-based)
        self.experts = nn.ModuleList([FeedForward(dim, mult=mult, dropout=dropout) for _ in range(num_experts)])

    def forward(self, x):
        B, N, D = x.shape
        tokens = x.reshape(B*N, D)

        logits = self.router(tokens)  # (T, E)
        probs = (logits / self.temperature).softmax(dim=-1)  # (T, E)

        if not self.hard:
            # soft mixture of ALL experts (more compute, but simplest + stable)
            out = 0
            for e, expert in enumerate(self.experts):
                out = out + expert(tokens) * probs[:, e:e+1]
            return out.view(B, N, D)

        topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(tokens)
        for e, expert in enumerate(self.experts):
            chosen = (topk_idx == e)
            if not chosen.any(): 
                continue
            token_ids, slot_ids = chosen.nonzero(as_tuple=True)
            expert_out = expert(tokens[token_ids])
            weights = topk_vals[token_ids, slot_ids].unsqueeze(-1)
            out[token_ids] += expert_out * weights
        return out.view(B, N, D)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None): #, atten_mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # pairwise attention mask: True=keep, False=block
        # if exists(atten_mask):
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     # allow (i, j), (b, i, j), or already (b*h, i, j)
        #     if atten_mask.dim() == 2:
        #         atten_mask = atten_mask.unsqueeze(0)                    # (1, i, j)
        #     if atten_mask.shape[0] == q.shape[0] // h:
        #         atten_mask = repeat(atten_mask, 'b i j -> (b h) i j', h=h)
        #     sim.masked_fill_(~atten_mask, max_neg_value)

        # attention
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        depth,
        num_freq_bands= None,
        max_freq= None,
        input_channels = 3,
        input_axis = 2,
        seq_dim = 77,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = None,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        final_classifier_head = True,
        use_moeffn = False,
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (e.g. 2 for images, 3 for video)
          seq_dim: The length of the input sequence (number of tokens), for images this would be the number of patches etc.
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
          use_moeffn: Whether to use Mixture of Experts FeedForward networks for latent blocks.
        """

        super().__init__()
        self.input_axis = input_axis
        self.seq_dim = seq_dim
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.fourier_encode_data = fourier_encode_data
        
        self.use_moeffn = use_moeffn
        # If fourier encoding, you must provide num_freq_bands and max_freq
        if fourier_encode_data:
            assert exists(num_freq_bands) and exists(max_freq), 'must provide num_freq_bands and max_freq if you are fourier encoding the data'
            fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
            input_dim = fourier_channels + input_channels
        else:
            input_dim = input_channels

        
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        with torch.no_grad():
            eps = 1e-6
            for n in range(num_latents // 2):
                i = n * 2
                j = n * 2 + 1
                vi = self.latents[i]
                vi = vi / (vi.norm() + eps)

                vj = self.latents[j]
                vj = vj - (vj @ vi) * vi
                vj = vj / (vj.norm() + eps)

                self.latents[i].copy_(vi)
                self.latents[j].copy_(vj)

        
        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        if not use_moeffn:
            get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        else:
            print(f'Initializing Perceiver with Mixture of Experts FeedForward layers with {num_latents} experts.')
            self.num_experts = num_latents
            # We will set top_k greater in the first layers and lower in the later layers, same for gate_do, forcing specialization in the later layers
            get_latent_ff = lambda top_k= 2, temp=1.0, gate_do=0.0: PreNorm(
                            latent_dim, 
                            MoEFeedForward(latent_dim,
                                    num_experts= self.num_experts,
                                    top_k=top_k,
                                    temperature=temp,
                                    dropout=ff_dropout,
                                    gate_dropout=gate_do
                                )
                            )

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}
            self_attns = nn.ModuleList([])

            if self.use_moeffn:
                # schedule: soft early -> hard late
                frac = i / max(depth - 1, 1)
                late = frac >= 0.6

                ff_top_k = 1 #2 if not late else 1          # soft-ish -> hard
                ff_temp  = 1.5 if not late else 0.6      # soft -> peaky
                ff_gdo   = 0.1 if not late else 0.0      # explore -> deterministic


                

                # append self attention blocks of latents
                for block_ind in range(self_per_cross_attn):
                    self_attns.append(nn.ModuleList([
                        get_latent_attn(**cache_args, key = block_ind),
                        get_latent_ff(top_k=ff_top_k, temp=ff_temp, 
                        gate_do=ff_gdo, **cache_args, key = (block_ind, ff_top_k, ff_temp, ff_gdo),)
                    ]))
            else:
                # append self attention blocks of latents
                for block_ind in range(self_per_cross_attn):
                    self_attns.append(nn.ModuleList([
                        get_latent_attn(**cache_args),
                        get_latent_ff(**cache_args)
                    ]))

            # for each layer, append cross attention between latents and inputs, 
            # followed by feedforward, then the self attention blocks of latents
            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        # classifier head that mean pools the latents and projects to number of classes (linear probing)
        if num_classes is not None:
            self.to_logits = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            ) if final_classifier_head else nn.Identity()
        else:
            self.to_logits = nn.Identity()



    def forward(
        self,
        data,
        mask = None,
        return_embeddings = False
        ):
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
            
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b = b)
            
            data = torch.cat((data, enc_pos), dim = -1)

        # concat to channels of data and flatten axis
        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b = b)
        
        b = x.shape[0]
        # atten_mask_keep = torch.tensor([
        #     [1,0,1,1,1,1],
        #     [0,1,1,1,1,1],
        #     [1,1,1,0,1,1],
        #     [1,1,0,1,1,1],
        #     [1,1,1,1,1,0],
        #     [1,1,1,1,0,1],
        # ], dtype=torch.bool).to(device)  # True where attention IS permitted
        # atten_mask_keep = atten_mask_keep.unsqueeze(0).expand(b, -1, -1)  # (b, 6, 6) 

        # layers
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x
            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x
        
        # allow for fetching embeddings

        if return_embeddings:
            return x

        # to logits

        return self.to_logits(x)