import torch
import numpy as np
from torch.utils.data import Dataset
# Adjust sys.path to import always from src
import os
import sys
import torch
from typing import Dict, Any, Callable, Optional, Sequence, Tuple, List, Union, Literal
import random
import re
from dataclasses import dataclass
import torch.nn as non

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from PIL import Image, ImageFilter

import torch
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode


# -----------------------------
# Common: CLIP normalization
# -----------------------------
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def make_gaussian_noise_transform(mean: float, sigma: float, clip: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    if hasattr(v2, "GaussianNoise"):
        return v2.GaussianNoise(mean=mean, sigma=sigma, clip=clip)

    def add_gaussian_noise(image: torch.Tensor) -> torch.Tensor:
        noisy = image + torch.randn_like(image) * sigma + mean
        if clip:
            noisy = noisy.clamp(0.0, 1.0)
        return noisy

    return add_gaussian_noise


# ============================================================
# 1) IMAGE AUGMENTATIONS (IRFL-safe: no hue/sat/color jitter)
# ============================================================
@dataclass(frozen=True)
class ImageAugConfig:
    size: int = 224
    # Mild crops so the key object are not cropped (esp. for color-dependent phrases)
    crop_scale: Tuple[float, float] = (0.95, 1.0)
    hflip_p: float = 0.95
    vflip_p: float = 0.95

    # blur parameters
    blur_kernel: int = 27


    # gaussian noise 
    mean: float = 0.0
    sigma: float = 0.1
    clip: bool = True

    # posterize image
    bits: int = 3
    post_p: float = 0.95

    # how many samples per "recipe"
    samples_per_recipe: int = 1



def make_image_augmentation_function(
    cfg: ImageAugConfig = ImageAugConfig(),
    return_tensors_normalized: bool = True,
    seed: Optional[int] = None,
) -> Callable[[Image.Image], List[Image.Image]]:
    """
    Returns a function augment_image(img) -> list of augmented views.
    - IRFL-safe: no hue/saturation jitter.
    - Produces a finite "menu" of augmentations (shared + unique recipes).
    """

    rng = random.Random(seed)

    # Base geometry transforms
    geom_hflip = transforms.Compose([
        transforms.RandomResizedCrop(cfg.size, scale=cfg.crop_scale,
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=cfg.hflip_p),
    ])
    geom_vflip = transforms.Compose([
        transforms.RandomResizedCrop(cfg.size, scale=cfg.crop_scale,
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomVerticalFlip(p=cfg.vflip_p),
    ])

    gaussian_blur = v2.GaussianBlur(kernel_size= cfg.blur_kernel)

    noise = make_gaussian_noise_transform(mean=cfg.mean, sigma=cfg.sigma, clip=cfg.clip)

    gaussian_noise = v2.Compose([
        v2.ToImage(),                           # PIL -> TVTensor
        v2.ToDtype(torch.float32, scale=True),  # [0,1]
        noise,
        v2.ToPILImage(),                        # back to PIL
    ])

    posterize = v2.RandomPosterize(bits=cfg.bits, p=cfg.post_p)

    def augment_image(img: Image.Image, augment_types: int) -> List[Image.Image]:
        if img.mode != "RGB":
            img0 = img.convert("RGB")
        else:
            img0 = img

        outs: List[Image.Image] = []

        
        chosen_types = rng.sample(range(5), k= augment_types)
        if 0 in chosen_types:
            # Horizontal flip and crop
            for _ in range(cfg.samples_per_recipe):
                v = geom_hflip(img0)
                outs.append(v)

        # Vertical flip and crop
        if 1 in chosen_types:
            for _ in range(cfg.samples_per_recipe):
                v = geom_vflip(img0)
                outs.append(v)

        # Gaussian Blur
        if 2 in chosen_types:
            for _ in range(cfg.samples_per_recipe):
                v = gaussian_blur(img0)
                outs.append(v)

        # Gaussian Noise
        if 3 in chosen_types:
            for _ in range(cfg.samples_per_recipe):
                v = gaussian_noise(img0)
                outs.append(v)

        # Posterize
        if 4 in chosen_types:
            for _ in range(cfg.samples_per_recipe):
                v = posterize(img0)
                outs.append(v)

        return outs

    return augment_image


# ============================================================
# 2) IDIOM/TEXT AUGMENTATIONS
# ============================================================
@dataclass(frozen=True)
class IdiomTextAugConfig:
    # neutral wrappers
    wrappers: Tuple[str, ...] = (
        "{t}",
        "text: {t}",
        "caption: {t}",
        "phrase: {t}"
    )
    # formatting-only variants
    add_quotes: bool = False
    add_period: bool = True
    add_exclamation: bool = False  # keep false by default
    # if True, returns unique strings only
    unique: bool = True


def make_text_augmentation_function(
    cfg: IdiomTextAugConfig = IdiomTextAugConfig(),
) -> Callable[[str], List[str]]:
    """
    For the figurative phrase itself (idiom/metaphor):
    - keeps the idiom unchanged
    - adds only neutral context / formatting (no "figurative", "idiom", etc.)
    """

    def augment_text(text: str) -> List[str]:
        t = " ".join(text.strip().split())  # normalize whitespace

        base = [w.format(t=t) for w in cfg.wrappers]

        return base

    return augment_text


# ============================================================
# 3) DEFINITION AUGMENTATIONS
# ============================================================
@dataclass(frozen=True)
class DefinitionAugConfig:
    wrappers: Tuple[str, ...] = (
        "{t}",
        "definition: {t}",
        "meaning: {t}",
        "explanation: {t}",
        "description: {t}",
    )
    # mild stochastic edits
    word_dropout_p: float = 0.08     # drop some non-critical tokens
    max_drops: int = 3
    swap_p: float = 0.15            # swap adjacent words a couple times
    max_swaps: int = 2
   
    protect_tokens_regex: str = r"^\d+$|^[A-Z]{2,}$"  # numbers, acronyms
    unique: bool = True
    seed: Optional[int] = None


_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _tokenize_simple(s: str) -> List[str]:
    return _WORD_RE.findall(s)


def _detokenize_simple(tokens: Sequence[str]) -> str:
    # naive detokenization: join with spaces, then fix common punctuation spacing
    s = " ".join(tokens)
    # remove space before punctuation
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    # fix opening quotes/brackets spacing
    s = re.sub(r"([(\[\{\"'])\s+", r"\1", s)
    # fix closing brackets spacing
    s = re.sub(r"\s+([)\]\}\"\'])", r"\1", s)
    return s.strip()


def make_definition_augmentation_function(
    cfg: DefinitionAugConfig = DefinitionAugConfig(),
) -> Callable[[str], List[str]]:
    rng = random.Random(cfg.seed)
    protect_re = re.compile(cfg.protect_tokens_regex)

    def _word_dropout(tokens: List[str]) -> List[str]:
        # drop only word-like tokens
        word_idxs = [
            i for i, tok in enumerate(tokens)
            if re.match(r"^\w+$", tok) and not protect_re.match(tok)
        ]
        if not word_idxs:
            return tokens

        drops = [i for i in word_idxs if rng.random() < cfg.word_dropout_p]
        drops = drops[: cfg.max_drops]
        if not drops:
            return tokens

        drop_set = set(drops)
        return [tok for i, tok in enumerate(tokens) if i not in drop_set]


    def augment_definition(text: str) -> List[str]:
        t0 = " ".join(text.strip().split())
        outs: List[str] = []

        # Base wrapped variants
        wrapped = [w.format(t=t0) for w in cfg.wrappers]

        outs.extend(wrapped)

        # Stochastic semantic-preserving edits
        for base in wrapped:
            toks = _tokenize_simple(base)

            # dropout view
            toks_d = _word_dropout(toks)
            outs.append(_detokenize_simple(toks_d))


        if cfg.unique:
            seen = set()
            deduped = []
            for s in outs:
                if s not in seen:
                    seen.add(s)
                    deduped.append(s)
            return deduped

        return outs

    return augment_definition
