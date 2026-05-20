import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
import torch.nn.functional as F
import typing
from typing import Literal, List
from src.models import repercent, jointopt
from posthoc.irfl.helper_metrics import test_fwd
import numpy as np
from einops import rearrange
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap

from posthoc.plotting_config import apply_paper_plot_style

apply_paper_plot_style()


def all_to_np(x):
    out = {}
    for key, val in x.items():
        if isinstance(val, list):
            out[key] = np.array(val)
        elif isinstance(val, dict):
            out[key] = {k: np.array(v) for k, v in val.items()}
        else:
            out[key] = val
    return out


def extract_all_embeddings(model, test_loader, device, M= 3, comp_mod= 1):
    embeds_all = {"Images": {"Unique": [], "Shared": []},
                    "Text": {"Unique": [], "Shared": []},
                    "figurative_types": []}

   
    model.eval()
    with torch.inference_mode():
        for batch_idx, out in enumerate(test_loader):
            print(f"Processing batch {batch_idx + 1}/ {len(test_loader)}")
            x = out['x']
            
            if comp_mod < 3:
                outputs = test_fwd(x, model, device, M= M)

                shared_text = outputs['S_view'][:, comp_mod, 0]
                shared_text = F.normalize(shared_text, dim=-1)

                unique_text = outputs['U'][:, comp_mod, 0]
                unique_text = F.normalize(unique_text, dim=-1)

                # shared_image_answers: [B, D]
                shared_image = outputs['S_view'][:, 0, comp_mod]
                shared_image = F.normalize(shared_image, dim=-1)

                unique_image = outputs['U'][:, 0, comp_mod]
                unique_image = F.normalize(unique_image, dim=-1)

            
            embeds_all["Images"]["Unique"] += unique_image.cpu().numpy().tolist()
            embeds_all["Images"]["Shared"] += shared_image.cpu().numpy().tolist()
            embeds_all["Text"]["Unique"] += unique_text.cpu().numpy().tolist()
            embeds_all["Text"]["Shared"] += shared_text.cpu().numpy().tolist()

            if "figurative_type" in out:
                ftypes = out["figurative_type"]
                embeds_all["figurative_types"] += ftypes
            
    
    return all_to_np(embeds_all)




def reduce_d(X, method="pca", dim: int= 2, random_state=0, **kwargs):
    """
    Reduce [N, D] -> [N, 2] using PCA, t-SNE, or UMAP.
    """
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=dim, random_state=random_state)
        return reducer.fit_transform(X)

    if method in {"tsne", "t-sne", "t_sne"}:
        n = X.shape[0]
        # t-SNE constraints: perplexity < n_samples
        perplexity = kwargs.pop("perplexity", min(70, max(2, (n - 1) // 3)))
        perplexity = min(perplexity, n - 1)  # ensure valid
        reducer = TSNE(
            n_components=dim,
            random_state=random_state,
            init="random",
            learning_rate="auto",
            perplexity=perplexity,
            method = "exact",
            **kwargs
        )
        return reducer.fit_transform(X)

    if method == "umap":
        # UMAP parameters with sensible defaults
        n_neighbors = kwargs.pop("n_neighbors", 80)
        min_dist = kwargs.pop("min_dist", 0.8)
        metric = kwargs.pop("metric", "euclidean")
        reducer = umap.UMAP(
            n_components=dim,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            **kwargs
        )
        return reducer.fit_transform(X)

    raise ValueError("method must be 'pca', 'tsne', or 'umap'")


def plot_embeddings(embeds_all, method="pca", f_type:Literal["all", "metaphor", "idiom", "simile"] = "all", random_state=0, fig_path:str=None, dim: int=2, **kwargs):
    """
    Plot the unique and shared embeddings for images and text, colored by modality type.
    Args:
        embeds_all: dict containing the embeddings and figurative types, as returned by extract_all_embeddings()
        method: "pca", "tsne", or "umap" for dimensionality reduction
        f_type: which figurative type to plot (default "all")
        random_state: random state for dimensionality reduction
        fig_path: if provided, save the figure to this path instead of showing it
        **kwargs: additional arguments for dimensionality reduction (e.g., perplexity for t-SNE, n_neighbors for UMAP)
    """
    
    
    # Reduce dimensions
    reduced_embeds = {}
    match f_type:
        case "all":
            u_images = embeds_all["Images"]["Unique"]
            s_images = embeds_all["Images"]["Shared"]
            u_text = embeds_all["Text"]["Unique"]
            s_text = embeds_all["Text"]["Shared"]
        
        case _:
            get_x_type = lambda x, f_types, f_type: [e for e, t in zip(x, f_types) if t == f_type]
            u_images = get_x_type(embeds_all["Images"]["Unique"], embeds_all["figurative_types"], f_type)
            s_images = get_x_type(embeds_all["Images"]["Shared"], embeds_all["figurative_types"], f_type)
            u_text = get_x_type(embeds_all["Text"]["Unique"], embeds_all["figurative_types"], f_type)
            s_text = get_x_type(embeds_all["Text"]["Shared"], embeds_all["figurative_types"], f_type)

    l_u_im, l_s_im, l_u_text, l_s_text = len(u_images), len(s_images), len(u_text), len(s_text)
    print(f"Number of points - Unique Images: {l_u_im}, Shared Images: {l_s_im}, Unique Text: {l_u_text}, Shared Text: {l_s_text}")

    concat = np.concatenate([u_images, s_images, u_text, s_text], axis=0)
    reduced_concat = reduce_d(concat, method=method, dim=dim, random_state=random_state, **kwargs)
    reduced_embeds["Images"] = {
        "Unique": reduced_concat[:l_u_im],
        "Shared": reduced_concat[l_u_im:l_u_im + l_s_im]
    }
    reduced_embeds["Text"] = {
        "Unique": reduced_concat[l_u_im + l_s_im:l_u_im + l_s_im + l_u_text],
        "Shared": reduced_concat[l_u_im + l_s_im + l_u_text:]
    }

    dim = reduced_embeds["Images"]["Unique"].shape[1]

    

    # Plotting
    colors = {
        "u_images": "skyblue",
        "s_images": "dodgerblue",
        "u_text": "lightcoral",
        "s_text": "red"
    }
    if dim == 3:
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(*reduced_embeds["Images"]["Unique"].T,
                   label="Unique Images", alpha=0.7, c=colors["u_images"])

        ax.scatter(*reduced_embeds["Images"]["Shared"].T,
                   label="Shared Images", alpha=0.7, c=colors["s_images"])

        ax.scatter(*reduced_embeds["Text"]["Unique"].T,
                   label="Unique Text", alpha=0.7, c=colors["u_text"])

        ax.scatter(*reduced_embeds["Text"]["Shared"].T,
                   label="Shared Text", alpha=0.7, c=colors["s_text"])
        
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")

    else:  # default to 2D
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(*reduced_embeds["Images"]["Unique"].T[:2],
                   label="Unique Images", alpha=0.7, c=colors["u_images"])

        ax.scatter(*reduced_embeds["Images"]["Shared"].T[:2],
                   label="Shared Images", alpha=0.7, c=colors["s_images"])

        ax.scatter(*reduced_embeds["Text"]["Unique"].T[:2],
                   label="Unique Text", alpha=0.7, c=colors["u_text"])

        ax.scatter(*reduced_embeds["Text"]["Shared"].T[:2],
                   label="Shared Text", alpha=0.7, c=colors["s_text"])

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
    ax.legend()
    if fig_path is not None:
        plt.savefig(fig_path)
    else:
        plt.show()

    return reduced_embeds
