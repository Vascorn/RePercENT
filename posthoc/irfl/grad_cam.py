import argparse
import os
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open_clip as clip
import torch
import yaml
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from torch.utils.data import DataLoader

from posthoc.plotting_config import apply_paper_plot_style
from src.models.repercent import RePercENT
from src.utils.helpers import set_seed
from src.utils.irfl_dataset import make_dataset
from training.train_repercent import make_model
import textwrap

apply_paper_plot_style()

GRAD_CAM_CMAP = "magma"
GRAD_CAM_ALPHA = 0.65
TEXT_BAR_COLOR = "#2B8CBE"
TEXT_BAR_EDGE_COLOR = "#045A8D"

def _normalize_cam(cam: torch.Tensor) -> np.ndarray:
    cam = torch.relu(cam)
    cam = cam - cam.min()
    den = cam.max() - cam.min()
    if den > 0:
        cam = cam / den
    return cam.detach().cpu().numpy()


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


def _forward_clip_preprojection(
    clip_model,
    image_input: torch.Tensor,
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    visual = clip_model.visual

    # Image branch: pre-projection patch features from CLIP ViT.
    x_img = image_input
    x_img = visual.conv1(x_img)  # [B, width, gh, gw]
    bsz, chn, gh, gw = x_img.shape
    x_img = x_img.reshape(bsz, chn, gh * gw).permute(0, 2, 1)  # [B, N, width]

    cls_t = visual.class_embedding
    cls_t = cls_t + torch.zeros(bsz, 1, x_img.shape[-1], device=x_img.device)
    x_img = torch.cat([cls_t, x_img], dim=1)  # [B, 1+N, width]

    x_img = x_img + visual.positional_embedding
    x_img = visual.ln_pre(x_img)
    x_img = x_img.permute(1, 0, 2)
    x_img = visual.transformer(x_img)
    x_img = x_img.permute(1, 0, 2)
    x_img = visual.ln_post(x_img)  # [B, 1+N, width], pre-projection

    patch_feats = x_img[:, 1:, :].float().detach().requires_grad_(True)

    # Text branch: pre-projection token features from CLIP text transformer.
    x_txt = clip_model.token_embedding(token_ids)
    x_txt = x_txt + clip_model.positional_embedding
    x_txt = x_txt.permute(1, 0, 2)
    x_txt = clip_model.transformer(x_txt)
    x_txt = x_txt.permute(1, 0, 2)
    x_txt = clip_model.ln_final(x_txt)  # [B, T, 512], pre-projection
    txt_feats = x_txt.float().detach().requires_grad_(True)

    txt_mask = token_ids != 0
    return patch_feats, txt_feats, txt_mask, token_ids, image_input


def _latent_scalar(outputs: Dict[str, torch.Tensor], target: str) -> torch.Tensor:
    targets = {
        "image_unique": outputs["U"][:, 0, 1, :],
        "image_shared": outputs["S_view"][:, 0, 1, :],
        "text_unique": outputs["U"][:, 1, 0, :],
        "text_shared": outputs["S_view"][:, 1, 0, :],
    }
    if target not in targets:
        raise ValueError(f"Unsupported target '{target}'")
    return (targets[target] ** 2).sum()


def _gradcam_1d(
    acts: torch.Tensor,
    grads: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> np.ndarray:
    """
    acts, grads: [B, T, C]
    mask: [B, T] with True for valid positions
    """
    if mask is None:
        weights = grads.mean(dim=1, keepdim=True)  # [B, 1, C]
        cam = (acts * weights).sum(dim=-1).squeeze(0)  # [T]
    else:
        mask_f = mask.unsqueeze(-1).float()  # [B, T, 1]
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1, 1]
        weights = (grads * mask_f).sum(dim=1, keepdim=True) / denom  # [B, 1, C]
        cam = ((acts * weights).sum(dim=-1) * mask.float()).squeeze(0)  # [T]

    return _normalize_cam(cam)


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


def _compute_pair_cams(
    model,
    clip_model,
    image_input: torch.Tensor,
    token_ids: torch.Tensor,
):
    results = {}

    for target_name in ["image_shared", "image_unique", "text_shared", "text_unique"]:
        model.zero_grad(set_to_none=True)

        patch_feats, txt_feats, txt_mask, tok_ids, img_in = _forward_clip_preprojection(
            clip_model=clip_model,
            image_input=image_input,
            token_ids=token_ids,
        )

        outputs = model([patch_feats, txt_feats], mask=[None, txt_mask.bool()])
        target_scalar = _latent_scalar(outputs, target_name)
        target_scalar.backward()

        if target_name.startswith("image_"):
            results[target_name] = _gradcam_1d(patch_feats, patch_feats.grad, mask=None)
        else:
            results[target_name] = _gradcam_1d(txt_feats, txt_feats.grad, mask=txt_mask.bool())

    return results, tok_ids, img_in


def _get_pair_panel(
    image_input: torch.Tensor,
    token_ids: torch.Tensor,
    tokenizer,
    cams: Dict[str, np.ndarray],
    phrase: str,
    pair_label: str,
) -> Dict[str, np.ndarray]:
    img = image_input.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    img = (img - img.min()) / max(img.max() - img.min(), 1e-8)

    image_shared_cam = cams["image_shared"]
    image_unique_cam = cams["image_unique"]
    text_shared_cam = cams["text_shared"]
    text_unique_cam = cams["text_unique"]

    side = int(np.sqrt(image_shared_cam.shape[0]))
    if side * side != image_shared_cam.shape[0]:
        raise ValueError(f"Number of image tokens must be a square, got {image_shared_cam.shape[0]}")

    shared_grid = image_shared_cam.reshape(side, side)
    unique_grid = image_unique_cam.reshape(side, side)

    labels = _token_labels(token_ids.squeeze(0), tokenizer)
    valid_len = int((token_ids.squeeze(0) != 0).sum().item())
    labels = labels[:valid_len]
    text_shared_scores = text_shared_cam[:valid_len]
    text_unique_scores = text_unique_cam[:valid_len]


    return {
        "image": img,
        "image_name": pair_label,
        "image_unique": unique_grid,
        "image_shared": shared_grid,
        "text_unique": (labels, text_unique_scores),
        "text_shared": (labels, text_shared_scores)
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
            cmap=GRAD_CAM_CMAP,
            alpha=GRAD_CAM_ALPHA,
            interpolation="bicubic",
            extent=[0, w, h, 0],
        )
        axes[row, 0].set_title(r"$s_{\mathrm{image}}$", fontsize=22, pad=12)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(image)
        axes[row, 1].imshow(
            image_unique,
            cmap=GRAD_CAM_CMAP,
            alpha=GRAD_CAM_ALPHA,
            interpolation="bicubic",
            extent=[0, w, h, 0],
        )
        axes[row, 1].set_title(r"$u_{\mathrm{image}}$", fontsize=22, pad=12)
        axes[row, 1].axis("off")

    # Use text gradients from the original pair only
    text_panel = panels[0]
    text_shared_labels, text_shared_scores = text_panel["text_shared"]
    text_unique_labels, text_unique_scores = text_panel["text_unique"]

    def prettify_labels(labels, width=18):
        return ["\n".join(textwrap.wrap(str(lbl), width=width)) for lbl in labels]

    shared_labels_pretty = prettify_labels(text_shared_labels, width=18)
    unique_labels_pretty = prettify_labels(text_unique_labels, width=18)

    bar_kwargs = dict(
        color=TEXT_BAR_COLOR,
        edgecolor=TEXT_BAR_EDGE_COLOR,
        linewidth=0.8,
        alpha=0.9
    )

    # Shared text
    x_shared = np.arange(len(text_shared_labels))
    axes[-1, 0].bar(x_shared, text_shared_scores, **bar_kwargs)
    axes[-1, 0].set_xticks(x_shared)
    axes[-1, 0].set_xticklabels(
        shared_labels_pretty,
        rotation=20,
        ha="right",
        rotation_mode="anchor",
        fontsize=15
    )
    axes[-1, 0].set_ylim(0.0, 1.0)
    axes[-1, 0].set_ylabel("Grad-CAM", fontsize=18)
    axes[-1, 0].set_title(r"$s_{\mathrm{text}}$", fontsize=22, pad=12)
    axes[-1, 0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[-1, 0].tick_params(axis="y", labelsize=14)

    # Unique text
    x_unique = np.arange(len(text_unique_labels))
    axes[-1, 1].bar(x_unique, text_unique_scores, **bar_kwargs)
    axes[-1, 1].set_xticks(x_unique)
    axes[-1, 1].set_xticklabels(
        unique_labels_pretty,
        rotation=20,
        ha="right",
        rotation_mode="anchor",
        fontsize=15
    )
    axes[-1, 1].set_ylim(0.0, 1.0)
    axes[-1, 1].set_ylabel("Grad-CAM", fontsize=18)
    axes[-1, 1].set_title(r"$u_{\mathrm{text}}$", fontsize=22, pad=12)
    axes[-1, 1].grid(axis="y", linestyle="--", alpha=0.3)
    axes[-1, 1].tick_params(axis="y", labelsize=14)

    fig.suptitle("Image-Text Pair Panels", fontsize=24, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    

def main():
    parser = argparse.ArgumentParser(
        description="For each (phrase, image) pair, save a 2x2 panel: image shared, image unique, text shared, text unique."
    )
    parser.add_argument("--datasets_path", type=str, default="../../data/irfl/datasets/", help="Path to IRFL dataset tensors wrt this script")
    
    parser.add_argument("--select_seed", type=int, default=0, help="Checkpoint seed index to load")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32", help="OpenCLIP model name")
    parser.add_argument("--clip_checkpoint", type=str, default="", help="Optional CLIP state_dict checkpoint path")
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

    test_data = torch.load(os.path.join(script_dir, args.datasets_path, "IRFL_train_tensors_2.pt"), map_location="cpu")
    test_data_aug = torch.load(os.path.join(script_dir, args.datasets_path, "IRFL_train_tensors_aug_2.pt"), map_location="cpu")
    test_dataset, _ = make_dataset(
        total_data=test_data | test_data_aug,
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


    for i, sample in enumerate(test_dataset):
        if i <= 2500:
            continue
        orig = sample["orig"]
        phrase = orig["phrases"]

        image_names = [
            ("original", orig["images"])#,
            # ("distractor_1", orig["distractors"][0]),
            # ("distractor_2", orig["distractors"][1]),
            # ("distractor_3", orig["distractors"][2]),
        ]

        token_ids = tokenizer([phrase]).to(device)

        out_dir = os.path.join(script_dir, "figures", "grad_cam")
        os.makedirs(out_dir, exist_ok=True)

        
        print(f"phrase='{phrase}'")
        print(f"pairs={image_names}")
        panels = [] # extract all panels first and then plot together
        for pair_label, image_name in image_names:
            pil_img = _load_image(image_name, project_root)
            image_input = preprocess(pil_img).unsqueeze(0).to(device)

            cams, tok_ids, img_in = _compute_pair_cams(
                model=model,
                clip_model=clip_model,
                image_input=image_input,
                token_ids=token_ids,
            )

            

            panel = _get_pair_panel(
                image_input=img_in,
                token_ids=tok_ids,
                tokenizer=tokenizer,
                cams=cams,
                phrase=phrase,
                pair_label=pair_label
            )

            panels.append(panel)


        out_path = os.path.join(
                out_dir,
                f"sample_{i}.pdf",
            )

        plot_all_pair_panels(panels, out_path)
        

if __name__ == "__main__":
    main()
