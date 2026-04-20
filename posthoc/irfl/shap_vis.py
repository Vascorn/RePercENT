import argparse
import os
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open_clip as clip
import torch
import torch.nn as nn
import yaml
from PIL import Image

try:
    import shap
except ImportError:
    shap = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from posthoc.plotting_config import apply_paper_plot_style
from src.models.repercent import RePercENT
from src.utils.helpers import set_seed
from src.utils.irfl_dataset import make_dataset
from training.train_repercent import make_model

apply_paper_plot_style()


def _normalize_attr(attr: np.ndarray) -> np.ndarray:
    attr = np.abs(attr)
    attr = attr - attr.min()
    den = attr.max() - attr.min()
    if den > 0:
        attr = attr / den
    return attr


def _token_labels(token_ids: torch.Tensor, tokenizer) -> list[str]:
    labels = []
    tok_list = token_ids.detach().cpu().tolist()
    for tid in tok_list:
        if tid == 0:
            labels.append("<pad>")
            continue
        if hasattr(tokenizer, "decode"):
            txt = tokenizer.decode([tid]).strip()
            labels.append(txt if txt else f"tok_{tid}")
        else:
            labels.append(f"tok_{tid}")
    return labels


def _clip_image_preprojection(clip_model, image_input: torch.Tensor) -> torch.Tensor:
    visual = clip_model.visual

    x_img = visual.conv1(image_input)
    bsz, chn, gh, gw = x_img.shape
    x_img = x_img.reshape(bsz, chn, gh * gw).permute(0, 2, 1)

    cls_t = visual.class_embedding
    cls_t = cls_t + torch.zeros(bsz, 1, x_img.shape[-1], device=x_img.device)
    x_img = torch.cat([cls_t, x_img], dim=1)

    x_img = x_img + visual.positional_embedding
    x_img = visual.ln_pre(x_img)
    x_img = x_img.permute(1, 0, 2)
    x_img = visual.transformer(x_img)
    x_img = x_img.permute(1, 0, 2)
    x_img = visual.ln_post(x_img)

    return x_img[:, 1:, :].float()


def _clip_text_preprojection(clip_model, text_embeddings: torch.Tensor) -> torch.Tensor:
    x_txt = text_embeddings + clip_model.positional_embedding
    x_txt = x_txt.permute(1, 0, 2)
    x_txt = clip_model.transformer(x_txt)
    x_txt = x_txt.permute(1, 0, 2)
    x_txt = clip_model.ln_final(x_txt)
    return x_txt.float()


def _target_spec(target: str) -> tuple[int, str]:
    specs = {
        "image_unique": (1, "U_12"),
        "image_shared": (1, "S_12"),
        "text_unique": (2, "U_21"),
        "text_shared": (2, "S_21"),
    }
    if target not in specs:
        raise ValueError(f"Unsupported target '{target}'")
    return specs[target]


def _modality_pos_enc(model, modality_idx: int) -> torch.Tensor | None:
    if not getattr(model, "add_pos_encoding", False):
        return None
    p_idx = getattr(model, f"pair_idx_m{modality_idx}")
    t_idx = getattr(model, f"type_idx_m{modality_idx}")
    pair_pe = model.pair_pos_enc[p_idx]
    type_pe = model.type_pos_enc[t_idx]
    return pair_pe + type_pe


class _SingleModalityTargetModel(nn.Module):
    def __init__(
        self,
        clip_model,
        model,
        modality_idx: int,
        comp_name: str,
        mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.model = model
        self.modality_idx = modality_idx
        self.comp_name = comp_name
        self.has_mask = mask is not None
        if self.has_mask:
            self.register_buffer("mask", mask.bool(), persistent=False)

    def forward(self, modality_input: torch.Tensor) -> torch.Tensor:
        mask = None
        if self.has_mask:
            mask = self.mask.expand(modality_input.shape[0], -1)

        if self.modality_idx == 1:
            feats = _clip_image_preprojection(self.clip_model, modality_input)
        elif self.modality_idx == 2:
            feats = _clip_text_preprojection(self.clip_model, modality_input)
        else:
            raise ValueError(f"Unsupported modality index '{self.modality_idx}'")

        pos_enc = _modality_pos_enc(self.model, self.modality_idx)
        encoded = self.model.disenEncoders[self.modality_idx - 1](feats, mask=mask, pos_enc=pos_enc)
        latents = self.model.get_slot(encoded, self.modality_idx, self.comp_name)
        return (latents ** 2).sum(dim=-1, keepdim=True)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _reduce_token_attr(attr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    attr = np.abs(attr)
    if attr.ndim < 2:
        raise ValueError(f"Expected attribution tensor with at least 2 dims, got shape {attr.shape}")

    reduce_axes = tuple(range(2, attr.ndim))
    if reduce_axes:
        scores = attr.mean(axis=reduce_axes)
    else:
        scores = attr

    if scores.ndim > 1:
        scores = scores.squeeze(0)

    if mask is not None:
        scores = scores * mask.squeeze(0).astype(scores.dtype)
    return _normalize_attr(scores)


def _reduce_image_attr(attr: np.ndarray, clip_model, image_input: torch.Tensor) -> np.ndarray:
    attr_t = torch.as_tensor(attr, device=image_input.device, dtype=image_input.dtype)

    if attr_t.ndim == 3:
        attr_t = attr_t.unsqueeze(0)
    elif attr_t.ndim > 4:
        spatial_shape = tuple(image_input.shape[-2:])
        channel_dim = image_input.shape[1]
        squeeze_dims = [dim for dim in range(attr_t.ndim) if attr_t.shape[dim] == 1]
        for dim in reversed(squeeze_dims):
            if attr_t.ndim <= 4:
                break
            attr_t = attr_t.squeeze(dim)
        if attr_t.ndim > 4:
            if attr_t.shape[1] == channel_dim and tuple(attr_t.shape[2:4]) == spatial_shape:
                trailing = tuple(range(4, attr_t.ndim))
                attr_t = attr_t.mean(dim=trailing)
            else:
                raise ValueError(f"Unsupported image attribution shape {tuple(attr_t.shape)}")

    if attr_t.ndim != 4:
        raise ValueError(f"Expected image attribution tensor with 4 dims, got shape {tuple(attr_t.shape)}")

    pixel_scores = attr_t.abs().mean(dim=1, keepdim=True)
    with torch.no_grad():
        gh, gw = clip_model.visual.conv1(image_input).shape[-2:]
    patch_scores = torch.nn.functional.adaptive_avg_pool2d(pixel_scores, (gh, gw))
    return _normalize_attr(patch_scores.squeeze(0).squeeze(0).detach().cpu().numpy().reshape(-1))


def _load_image(image_name: str, project_root: str) -> Image.Image:
    image_path = os.path.join(project_root, "data", "irfl", "images", image_name.split(".")[0] + ".jpeg")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def build_repercent(model_config, data_config, device: torch.device) -> RePercENT:
    m = data_config["create_data"]["M"]
    disen_encoders = [make_model(model_config, data_config, modality=i + 1, M=m) for i in range(m)]
    model = RePercENT(
        M=m,
        disenEncoder=disen_encoders,
        disen_mapping=model_config["repercent"]["disen_mapping"],
        vmfkappa=model_config["repercent"]["vmfkappa"],
    ).to(device)
    return model


def _compute_pair_shap(
    model,
    clip_model,
    image_input: torch.Tensor,
    token_ids: torch.Tensor,
    nsamples: int,
):
    if shap is None:
        raise ImportError(
            "The 'shap' package is not installed. Install it with `pip install shap` to run SHAP visualizations."
        )

    results = {}
    txt_mask = token_ids != 0
    txt_embed_input = clip_model.token_embedding(token_ids).float()
    image_bg = torch.zeros_like(image_input)
    txt_bg = torch.zeros_like(txt_embed_input)
    txt_mask_np = _to_numpy(txt_mask)

    for target_name in ["image_shared", "image_unique", "text_shared", "text_unique"]:
        modality_idx, comp_name = _target_spec(target_name)
        if modality_idx == 1:
            wrapped_model = _SingleModalityTargetModel(
                clip_model=clip_model,
                model=model,
                modality_idx=modality_idx,
                comp_name=comp_name,
            )
            explainer = shap.GradientExplainer(wrapped_model, image_bg)
            attr = explainer.shap_values(image_input, nsamples=nsamples)
            attr = _to_numpy(attr)
            results[target_name] = _reduce_image_attr(attr, clip_model=clip_model, image_input=image_input)
        else:
            wrapped_model = _SingleModalityTargetModel(
                clip_model=clip_model,
                model=model,
                modality_idx=modality_idx,
                comp_name=comp_name,
                mask=txt_mask,
            )
            explainer = shap.GradientExplainer(wrapped_model, txt_bg)
            attr = explainer.shap_values(txt_embed_input, nsamples=nsamples)
            attr = _to_numpy(attr)
            results[target_name] = _reduce_token_attr(attr, mask=txt_mask_np)

    return results, token_ids, image_input


def _get_pair_panel(
    image_input: torch.Tensor,
    token_ids: torch.Tensor,
    tokenizer,
    shap_values: Dict[str, np.ndarray],
    pair_label: str,
) -> Dict[str, np.ndarray]:
    img = image_input.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / max(img.max() - img.min(), 1e-8)

    image_shared_attr = shap_values["image_shared"]
    image_unique_attr = shap_values["image_unique"]
    text_shared_attr = shap_values["text_shared"]
    text_unique_attr = shap_values["text_unique"]

    side = int(np.sqrt(image_shared_attr.shape[0]))
    if side * side != image_shared_attr.shape[0]:
        raise ValueError(f"Number of image tokens must be a square, got {image_shared_attr.shape[0]}")

    shared_grid = image_shared_attr.reshape(side, side)
    unique_grid = image_unique_attr.reshape(side, side)

    labels = _token_labels(token_ids.squeeze(0), tokenizer)
    valid_len = int((token_ids.squeeze(0) != 0).sum().item())
    labels = labels[:valid_len]
    text_shared_scores = text_shared_attr[:valid_len]
    text_unique_scores = text_unique_attr[:valid_len]

    return {
        "image": img,
        "image_name": pair_label,
        "image_unique": unique_grid,
        "image_shared": shared_grid,
        "text_unique": (labels, text_unique_scores),
        "text_shared": (labels, text_shared_scores),
    }


def plot_all_pair_panels(panels: list[Dict[str, np.ndarray]], out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    for row, panel in enumerate(panels):
        image = panel["image"]
        image_unique = panel["image_unique"]
        image_shared = panel["image_shared"]

        h, w = image.shape[:2]

        axes[row, 0].imshow(image)
        axes[row, 0].imshow(
            image_shared,
            cmap="jet",
            alpha=0.75,
            interpolation="bicubic",
            extent=[0, w, h, 0],
        )
        axes[row, 0].set_title(f"{panel['image_name']} shared wrt text")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(image)
        axes[row, 1].imshow(
            image_unique,
            cmap="jet",
            alpha=0.75,
            interpolation="bicubic",
            extent=[0, w, h, 0],
        )
        axes[row, 1].set_title(f"{panel['image_name']} unique wrt text")
        axes[row, 1].axis("off")

    text_panel = panels[0]
    text_shared_labels, text_shared_scores = text_panel["text_shared"]
    text_unique_labels, text_unique_scores = text_panel["text_unique"]

    axes[-1, 0].bar(np.arange(len(text_shared_labels)), text_shared_scores, color="tomato")
    axes[-1, 0].set_xticks(np.arange(len(text_shared_labels)))
    axes[-1, 0].set_xticklabels(text_shared_labels, rotation=70, ha="right", fontsize=8)
    axes[-1, 0].set_ylim(0.0, 1.0)
    axes[-1, 0].set_ylabel("SHAP")
    axes[-1, 0].set_title("Text shared SHAP distribution")

    axes[-1, 1].bar(np.arange(len(text_unique_labels)), text_unique_scores, color="tomato")
    axes[-1, 1].set_xticks(np.arange(len(text_unique_labels)))
    axes[-1, 1].set_xticklabels(text_unique_labels, rotation=70, ha="right", fontsize=8)
    axes[-1, 1].set_ylim(0.0, 1.0)
    axes[-1, 1].set_ylabel("SHAP")
    axes[-1, 1].set_title("Text unique SHAP distribution")

    fig.suptitle("Image-Text Pair Panels", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="For each (phrase, image) pair, save a 2x2 SHAP panel: image shared, image unique, text shared, text unique."
    )
    parser.add_argument("--datasets_path", type=str, default="../../data/irfl/datasets/", help="Path to IRFL dataset tensors wrt this script")
    parser.add_argument("--select_seed", type=int, default=0, help="Checkpoint seed index to load")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32", help="OpenCLIP model name")
    parser.add_argument("--clip_checkpoint", type=str, default="", help="Optional CLIP state_dict checkpoint path")
    parser.add_argument("--start_idx", type=int, default=0, help="Dataset start index")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of samples to render (-1 for all)")
    parser.add_argument("--shap_nsamples", type=int, default=128, help="Number of samples used by SHAP GradientExplainer")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    m = 2
    set_seed(2)

    data_config_path = os.path.join(project_root, "configs", "data", f"irfl_data_{m}m.yaml")
    model_config_path = os.path.join(project_root, "configs", "model", f"repercent_irfl_{m}m.yaml")
    analysis_config_path = os.path.join(project_root, "configs", "posthoc_analysis", f"irfl_{m}m.yaml")

    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    with open(analysis_config_path, "r") as f:
        analysis_config = yaml.safe_load(f)

    train_data = torch.load(os.path.join(script_dir, args.datasets_path, "IRFL_train_tensors_2.pt"), map_location="cpu")
    train_data_aug = torch.load(os.path.join(script_dir, args.datasets_path, "IRFL_train_tensors_aug_2.pt"), map_location="cpu")
    train_dataset, _ = make_dataset(
        total_data=train_data | train_data_aug,
        num_modalities=data_config["create_data"]["M"],
        data_type="train",
        include_original=True,
    )

    ckpt_rel = analysis_config["repercent"]["checkpoints"][args.select_seed]
    repercent_ckpt = os.path.join(project_root, ckpt_rel)

    model = build_repercent(model_config, data_config, device)
    model_state = torch.load(repercent_ckpt, map_location=device)
    model.load_state_dict(model_state["model_state_dict"])
    model.eval()

    clip_model, _, preprocess = clip.create_model_and_transforms(args.clip_model, pretrained="openai", device=device)
    tokenizer = clip.get_tokenizer(args.clip_model)
    if args.clip_checkpoint:
        clip_state = torch.load(args.clip_checkpoint, map_location=device)
        clip_model.load_state_dict(clip_state, strict=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    out_dir = os.path.join(script_dir, "figures", "shap")
    os.makedirs(out_dir, exist_ok=True)

    rendered = 0
    for i, sample in enumerate(train_dataset):
        if i < args.start_idx:
            continue
        if args.max_samples >= 0 and rendered >= args.max_samples:
            break

        orig = sample["orig"]
        phrase = orig["phrases"]
        image_names = [("original", orig["images"])]

        token_ids = tokenizer([phrase]).to(device)

        print(f"phrase='{phrase}'")
        print(f"pairs={image_names}")

        panels = []
        for pair_label, image_name in image_names:
            pil_img = _load_image(image_name, project_root)
            image_input = preprocess(pil_img).unsqueeze(0).to(device)

            shap_values, tok_ids, img_in = _compute_pair_shap(
                model=model,
                clip_model=clip_model,
                image_input=image_input,
                token_ids=token_ids,
                nsamples=args.shap_nsamples,
            )

            panel = _get_pair_panel(
                image_input=img_in,
                token_ids=tok_ids,
                tokenizer=tokenizer,
                shap_values=shap_values,
                pair_label=pair_label,
            )
            panels.append(panel)

        out_path = os.path.join(out_dir, f"sample_{i}.pdf")
        plot_all_pair_panels(panels, out_path)
        rendered += 1


if __name__ == "__main__":
    main()
