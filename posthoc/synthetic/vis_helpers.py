from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_pairwise_component_projections(data_dict, M: int, method: str = "pca", max_points: int = 3000, seed: int = 0):
    """
    Create one 2D projection plot per modality pair using latent components
    [u_ij, u_ji, s_ij].

    Args:
        data_dict: output of extract_latents_and_labels(...)
        M: number of modalities
        method: currently supports "pca" and "tsne"
        max_points: cap samples per component for readability
        seed: deterministic subsampling seed

    Returns:
        Dict[(i, j), matplotlib.figure.Figure] with 1-based modality indices.
    """
    
    figures = {}
    rng = np.random.default_rng(seed)

    for i, j in combinations(range(M), 2):
        u_ij = data_dict["U"][i][j]
        u_ji = data_dict["U"][j][i]
        s_ij = data_dict["S"][i][j]
        s_ji = data_dict["S"][j][i]

        if u_ij is None or u_ji is None or s_ij is None or s_ji is None:
            continue

        def _subsample(X):
            if X.shape[0] <= max_points:
                return X
            idx = rng.choice(X.shape[0], size=max_points, replace=False)
            return X[idx]

        u_ij_s = _subsample(u_ij)
        u_ji_s = _subsample(u_ji)
        s_ij_s = _subsample(s_ij)
        s_ji_s = _subsample(s_ji)

        Z = np.concatenate([u_ij_s, u_ji_s, s_ij_s, s_ji_s], axis=0)
        labels = np.array(
            [f"u_{i+1}{j+1}"] * u_ij_s.shape[0]
            + [f"u_{j+1}{i+1}"] * u_ji_s.shape[0]
            + [f"s_{i+1}{j+1}"] * s_ij_s.shape[0]
            + [f"s_{j+1}{i+1}"] * s_ji_s.shape[0]
        )
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

        if method.lower() == "pca":
            Z_2d = PCA(n_components=2, random_state=seed).fit_transform(Z)
            ax.set_xlabel("PCA1")
            ax.set_ylabel("PCA2")
        elif method.lower() == "tsne":
            Z_2d = TSNE(n_components=2, random_state=seed).fit_transform(Z)
            ax.set_xlabel("TSNE1")
            ax.set_ylabel("TSNE2")

        
        for comp_name in [f"u_{i+1}{j+1}", f"u_{j+1}{i+1}", f"s_{i+1}{j+1}", f"s_{j+1}{i+1}"]:
            mask = labels == comp_name
            ax.scatter(
                Z_2d[mask, 0],
                Z_2d[mask, 1],
                s=10,
                alpha=0.5,
                label=comp_name,
            )

        ax.set_title(f"2D Projection of Components for Pair ({i+1}, {j+1})")
        
        ax.legend(loc="best")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        figures[(i + 1, j + 1)] = fig

    return figures
