import argparse
import os
import sys
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange, repeat
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from posthoc.plotting_config import apply_paper_plot_style
from src.models.perceiver import SlotAttention, fourier_encode
from src.models.repercent import RePercENT
from src.utils.helpers import set_seed
from src.utils.irfl_dataset import make_dataset
from training.train_repercent import make_model


apply_paper_plot_style()


def _as_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Expected 'true' or 'false', got {value}")


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.min()
    denom = values.max() - values.min()
    if denom > 0:
        values = values / denom
    return values


def _contrast_attention(
    values: np.ndarray,
    *,
    subtract_uniform: bool = True,
    percentile: float = 92.0,
    gamma: float = 1.8,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if subtract_uniform:
        values = values - (1.0 / max(1, values.size))
        values = np.maximum(values, 0.0)

    hi = np.percentile(values, percentile)
    if hi > 0:
        values = np.clip(values / hi, 0.0, 1.0)
    else:
        values = _normalize_positive(values)

    if gamma != 1.0:
        values = values ** gamma
    return values


def _text_for_display(value) -> str:
    if isinstance(value, (list, tuple)):
        return ". ".join(str(v) for v in value)
    return str(value)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _load_image(image_name, project_root: str) -> Image.Image:
    if isinstance(image_name, (list, tuple)):
        image_name = image_name[0]
    image_path = os.path.join(
        project_root,
        "data",
        "irfl",
        "images",
        str(image_name).split(".")[0] + ".jpeg",
    )
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def build_repercent(
    model_config: dict,
    data_config: dict,
    device: torch.device,
    add_pos_encoding: bool,
) -> RePercENT:
    m = data_config["create_data"]["M"]
    disen_encoders = [
        make_model(model_config, data_config, modality=i + 1, M=m)
        for i in range(m)
    ]
    return RePercENT(
        M=m,
        disenEncoder=disen_encoders,
        disen_mapping=model_config["repercent"]["disen_mapping"],
        vmfkappa=model_config["repercent"]["vmfkappa"],
        add_pos_encoding=add_pos_encoding,
    ).to(device)


def _checkpoint_has_repercent_pos_encoding(state_dict: dict) -> bool:
    return "pair_pos_enc" in state_dict and "type_pos_enc" in state_dict


def _modality_pos_enc(model: RePercENT, modality_idx: int) -> torch.Tensor | None:
    if not getattr(model, "add_pos_encoding", False):
        return None
    pair_idx = getattr(model, f"pair_idx_m{modality_idx}")
    type_idx = getattr(model, f"type_idx_m{modality_idx}")
    return model.pair_pos_enc[pair_idx] + model.type_pos_enc[type_idx]


def _prepare_perceiver_context(perceiver, data: torch.Tensor) -> torch.Tensor:
    batch_size, *axis, _, device, dtype = *data.shape, data.device, data.dtype
    if len(axis) != perceiver.input_axis:
        raise ValueError(
            f"Expected input_axis={perceiver.input_axis}, got data shape {tuple(data.shape)}"
        )

    if perceiver.fourier_encode_data:
        axis_pos = [
            torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=dtype)
            for size in axis
        ]
        pos = torch.stack(torch.meshgrid(*axis_pos, indexing="ij"), dim=-1)
        enc_pos = fourier_encode(pos, perceiver.max_freq, perceiver.num_freq_bands)
        enc_pos = rearrange(enc_pos, "... n d -> ... (n d)")
        enc_pos = repeat(enc_pos, "... -> b ...", b=batch_size)
        data = torch.cat((data, enc_pos), dim=-1)

    return rearrange(data, "b ... d -> b (...) d")


def _compute_cross_attention(attn_module, x: torch.Tensor, context: torch.Tensor, mask=None):
    heads = attn_module.heads
    q = attn_module.to_q(x)
    k, _ = attn_module.to_kv(context).chunk(2, dim=-1)
    q, k = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=heads), (q, k))
    sim = torch.einsum("b i d, b j d -> b i j", q, k) * attn_module.scale

    if mask is not None:
        mask = rearrange(mask, "b ... -> b (...)").bool()
        mask = repeat(mask, "b j -> (b h) () j", h=heads)
        sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

    if isinstance(attn_module, SlotAttention):
        attn = attn_module._grouped_softmax_over_queries(sim, group_size=2)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
    else:
        attn = sim.softmax(dim=-1)

    batch_size = x.shape[0]
    return rearrange(attn, "(b h) i j -> b h i j", b=batch_size, h=heads)


def _encoder_cross_attention(
    model: RePercENT,
    modality_idx: int,
    data: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    encoder = model.disenEncoders[modality_idx - 1]
    perceiver = encoder.perceiver
    data = encoder.encoder(data)
    context = _prepare_perceiver_context(perceiver, data)

    batch_size = context.shape[0]
    x = repeat(perceiver.latents, "n d -> b n d", b=batch_size)
    pos_enc = _modality_pos_enc(model, modality_idx)
    if pos_enc is not None:
        x = x + pos_enc.unsqueeze(0)

    captured = []
    for idx, layer in enumerate(perceiver.layers):
        if len(layer) == 3:
            cross_attn, cross_ff, self_attns = layer
        else:
            cross_attn, cross_ff = layer
            self_attns = []

        norm_x = cross_attn.norm(x)
        norm_context = cross_attn.norm_context(context)
        captured.append(
            _compute_cross_attention(
                cross_attn.fn,
                norm_x,
                norm_context,
                mask=mask,
            )
        )

        if isinstance(cross_attn.fn, SlotAttention):
            x = cross_attn(x, context=context, mask=mask, group_size=2) + x
        else:
            x = cross_attn(x, context=context, mask=mask) + x
        x = cross_ff(x) + x

        for self_attn, self_ff in self_attns:
            x = self_attn(x) + x
            x = self_ff(x) + x

    if not captured:
        raise RuntimeError("Failed to capture cross-attention.")
    return torch.stack(captured, dim=0).mean(dim=0)


def component_token_attention(
    model: RePercENT,
    modality_idx: int,
    data: torch.Tensor,
    mask: torch.Tensor | None,
    components: Iterable[str],
) -> Dict[str, np.ndarray]:
    attention = _encoder_cross_attention(
        model=model,
        modality_idx=modality_idx,
        data=data,
        mask=mask,
    )
    attention = attention.mean(dim=1)

    out = {}
    mapping = model.disen_mapping[f"M_{modality_idx}"]
    for component in components:
        slot_idx = mapping[component]
        out[component] = attention[:, slot_idx, :].detach().cpu().numpy()
    return out


def encode_components(
    model: RePercENT,
    modality_idx: int,
    data: torch.Tensor,
    mask: torch.Tensor | None,
    components: Iterable[str],
) -> Dict[str, torch.Tensor]:
    encoded = model.disenEncoders[modality_idx - 1](
        data,
        mask=mask,
        pos_enc=_modality_pos_enc(model, modality_idx),
    )
    return {
        component: model.get_slot(encoded, modality_idx, component)
        for component in components
    }


def shared_retrieval_scores(
    model: RePercENT,
    image_candidates: torch.Tensor,
    caption_embeddings: torch.Tensor,
    caption_mask: torch.Tensor,
) -> torch.Tensor:
    image_components = encode_components(
        model=model,
        modality_idx=1,
        data=image_candidates,
        mask=None,
        components=("S_12",),
    )
    caption_components = encode_components(
        model=model,
        modality_idx=2,
        data=caption_embeddings,
        mask=caption_mask,
        components=("S_21",),
    )

    shared_images = F.normalize(image_components["S_12"], dim=-1)
    shared_caption = F.normalize(caption_components["S_21"], dim=-1)
    return (shared_caption @ shared_images.T).squeeze(0)


def plot_image_candidate_attention_panel(
    images: list[Image.Image],
    labels: list[str],
    scores: np.ndarray,
    shared_attention: np.ndarray,
    unique_attention: np.ndarray,
    phrase: str,
    out_path: str,
    percentile: float,
    gamma: float,
):
    n_candidates = len(images)
    if shared_attention.shape[0] != n_candidates or unique_attention.shape[0] != n_candidates:
        raise ValueError("Attention arrays must have one row per image candidate.")

    side = int(np.sqrt(shared_attention.shape[1]))
    if side * side != shared_attention.shape[1]:
        raise ValueError(
            f"Expected square image patch grid, got {shared_attention.shape[1]} patches"
        )

    fig, axes = plt.subplots(
        n_candidates,
        3,
        figsize=(11.5, 2.8 * n_candidates),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, (image, label, score) in enumerate(zip(images, labels, scores)):
        image_arr = np.asarray(image)
        width, height = image.size

        shared_grid = _contrast_attention(
            shared_attention[row_idx],
            percentile=percentile,
            gamma=gamma,
        ).reshape(side, side)
        unique_grid = _contrast_attention(
            unique_attention[row_idx],
            percentile=percentile,
            gamma=gamma,
        ).reshape(side, side)

        row_title = f"{label} | score={score:.3f}"
        axes[row_idx, 0].imshow(image_arr)
        axes[row_idx, 0].set_title(row_title)

        axes[row_idx, 1].imshow(image_arr)
        axes[row_idx, 1].imshow(
            shared_grid,
            cmap="magma",
            alpha=np.clip(shared_grid, 0.0, 1.0) * 0.85,
            interpolation="bicubic",
            extent=[0, width, height, 0],
            vmin=0.0,
            vmax=1.0,
        )
        axes[row_idx, 1].set_title(r"$S_{12}$")

        axes[row_idx, 2].imshow(image_arr)
        axes[row_idx, 2].imshow(
            unique_grid,
            cmap="magma",
            alpha=np.clip(unique_grid, 0.0, 1.0) * 0.85,
            interpolation="bicubic",
            extent=[0, width, height, 0],
            vmin=0.0,
            vmax=1.0,
        )
        axes[row_idx, 2].set_title(r"$U_{12}$")

        for ax in axes[row_idx]:
            ax.axis("off")

    fig.suptitle(phrase, fontsize=13)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Render RePercENT IRFL image component attention maps for correctly retrieved test samples."
    )
    parser.add_argument("--datasets_path", type=str, default="../../data/irfl/datasets/")
    parser.add_argument("--dataset_suffix", type=str, default="_2")
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--select_seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--num_distractors", type=int, default=3)
    parser.add_argument("--add_pos_encoding", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--use_slot_attn", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--out_dir", type=str, default="figures/attention_maps")
    parser.add_argument("--format", choices=["pdf", "png"], default="pdf")
    parser.add_argument("--attention_percentile", type=float, default=92.0)
    parser.add_argument("--attention_gamma", type=float, default=1.8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    set_seed(2)

    data_config_path = os.path.join(project_root, "configs", "data", f"irfl_data_{args.m}m.yaml")
    model_config_path = os.path.join(project_root, "configs", "model", f"repercent_irfl_{args.m}m.yaml")
    analysis_config_path = os.path.join(project_root, "configs", "posthoc_analysis", f"irfl_{args.m}m.yaml")

    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    with open(analysis_config_path, "r") as f:
        analysis_config = yaml.safe_load(f)

    if args.use_slot_attn != "auto":
        model_config["perceiver"]["use_slot_attn"] = _as_bool(args.use_slot_attn)

    ckpt_rel = analysis_config["repercent"]["checkpoints"][args.select_seed]
    ckpt_path = os.path.join(project_root, ckpt_rel)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    if args.add_pos_encoding == "auto":
        add_pos_encoding = _checkpoint_has_repercent_pos_encoding(state_dict)
    else:
        add_pos_encoding = _as_bool(args.add_pos_encoding)

    model = build_repercent(
        model_config=model_config,
        data_config=data_config,
        device=device,
        add_pos_encoding=add_pos_encoding,
    )
    model.load_state_dict(state_dict)
    model.eval()

    data_file = f"IRFL_test_tensors{args.dataset_suffix}.pt"
    aug_file = f"IRFL_test_tensors_aug{args.dataset_suffix}.pt"
    data = torch.load(os.path.join(script_dir, args.datasets_path, data_file), map_location="cpu")
    data_aug = torch.load(os.path.join(script_dir, args.datasets_path, aug_file), map_location="cpu")
    dataset, _ = make_dataset(
        total_data=data | data_aug,
        num_modalities=data_config["create_data"]["M"],
        data_type="test",
        include_original=True,
    )

    out_dir = os.path.join(script_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    examined = 0
    correct = 0
    rendered = 0
    for sample_idx, sample in enumerate(dataset):
        if rendered >= args.max_samples:
            break

        examined += 1
        x = sample["x"]
        orig = sample["orig"]
        phrase = _text_for_display(orig["phrases"])

        answer_embeddings = x["images"].unsqueeze(0).to(device)
        distractor_embeddings = sample["distractors"][: args.num_distractors].to(device).float()
        image_candidates = torch.cat([answer_embeddings, distractor_embeddings], dim=0)
        caption_embeddings = x["texts"].unsqueeze(0).to(device)
        caption_mask = x["pad_masks"].unsqueeze(0).bool().to(device)

        with torch.no_grad():
            scores = shared_retrieval_scores(
                model=model,
                image_candidates=image_candidates,
                caption_embeddings=caption_embeddings,
                caption_mask=caption_mask,
            )
            is_correct = bool(scores[0] > scores[1:].max())
            if not is_correct:
                continue
            correct += 1

            image_attn = component_token_attention(
                model=model,
                modality_idx=1,
                data=image_candidates,
                mask=None,
                components=("S_12", "U_12"),
            )

        distractor_names = _as_list(orig.get("distractors"))[: args.num_distractors]
        if len(distractor_names) < args.num_distractors:
            print(f"Skipping sample {sample_idx}: expected {args.num_distractors} distractors.")
            continue

        images = [_load_image(orig["images"], project_root)]
        images.extend(_load_image(name, project_root) for name in distractor_names)
        labels = ["Answer"] + [f"Distractor {idx + 1}" for idx in range(len(distractor_names))]
        out_path = os.path.join(out_dir, f"test_correct_sample_{sample_idx}.{args.format}")

        plot_image_candidate_attention_panel(
            images=images,
            labels=labels,
            scores=scores.detach().cpu().numpy(),
            shared_attention=image_attn["S_12"],
            unique_attention=image_attn["U_12"],
            phrase=phrase,
            out_path=out_path,
            percentile=args.attention_percentile,
            gamma=args.attention_gamma,
        )

        print(f"Saved {out_path}")
        rendered += 1

    print(
        f"Rendered {rendered} correctly retrieved samples "
        f"after examining {examined} test samples ({correct} correct encountered)."
    )


if __name__ == "__main__":
    main()
