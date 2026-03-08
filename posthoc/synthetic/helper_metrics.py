import re
from typing import Dict, Tuple, List, Optional

import numpy as np


_PAIR_RE = re.compile(r"^([us])_(\d+)(\d+)$|^([us])_(\d+)_(\d+)$")


def _parse_key(key: str) -> Tuple[str, int, int]:
    m = _PAIR_RE.match(key)
    if m is None:
        raise ValueError(f"Unsupported component key format: {key}")

    comp_type = m.group(1) or m.group(4)
    i = int(m.group(2) or m.group(5))
    j = int(m.group(3) or m.group(6))
    return comp_type, i, j


def _shared_key_for_direction(i: int, j: int, index_map: Dict[str, int]) -> Optional[str]:
    directed = [f"s_{i}{j}", f"s_{i}_{j}"]
    for key in directed:
        if key in index_map:
            return key

    # Fallback for codepaths where shared is stored once per unordered pair.
    a, b = min(i, j), max(i, j)
    canonical = [f"s_{a}{b}", f"s_{a}_{b}"]
    for key in canonical:
        if key in index_map:
            return key

    return None


def linear_probe_disentanglement_metric(
    linear_probe_acc: Dict[str, np.ndarray],
    expected_modalities: Optional[int] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute a pairwise disentanglement score from linear probe accuracies.

    Intended form (directed expectation over i != j):
      E[ u_ij -> u_ij + s_ij -> s_ij - s_ij -> u_ij - u_ij -> s_ij ]

    Args:
        linear_probe_acc:
            Dict[label_key -> accuracy array over component keys], typically
            linear_results["acc"] from ProbeEvaluator.calculate_linear_probe().
        expected_modalities:
            Optional sanity check for number of modalities M.
        eps:
            Small constant to avoid division by zero.

    Returns:
        Dictionary with raw and normalized variants.
    """
    component_keys: List[str] = list(linear_probe_acc.keys())
    index_map = {k: idx for idx, k in enumerate(component_keys)}

    # Collect all directed unique keys u_ij.
    unique_triplets: List[Tuple[int, int, str]] = []
    for key in component_keys:
        comp_type, i, j = _parse_key(key)
        if comp_type == "u":
            unique_triplets.append((i, j, key))

    if len(unique_triplets) == 0:
        raise ValueError("No unique component keys found (expected keys like 'u_12').")

    if expected_modalities is not None:
        modalities = set()
        for i, j, _ in unique_triplets:
            modalities.add(i)
            modalities.add(j)
        if len(modalities) != expected_modalities:
            raise ValueError(
                f"expected_modalities={expected_modalities}, but inferred {len(modalities)} from unique keys."
            )

    uu_vals: List[float] = []
    ss_vals: List[float] = []
    su_vals: List[float] = []
    us_vals: List[float] = []

    for i, j, u_key in unique_triplets:
        s_key = _shared_key_for_direction(i, j, index_map)
        if s_key is None:
            continue

        u_col = index_map[u_key]
        s_col = index_map[s_key]

        # u_ij -> u_ij
        uu_vals.append(float(linear_probe_acc[u_key][u_col]))
        # s_ij -> s_ij
        ss_vals.append(float(linear_probe_acc[s_key][s_col]))
        # s_ij -> u_ij
        su_vals.append(float(linear_probe_acc[s_key][u_col]))
        # u_ij -> s_ij
        us_vals.append(float(linear_probe_acc[u_key][s_col]))

    if min(len(uu_vals), len(ss_vals), len(su_vals), len(us_vals)) == 0:
        raise ValueError(
            "Insufficient cross-component entries to compute metric. "
            "Check that linear_probe_acc contains both u_ij and matching s_ij labels/components."
        )

    uu = float(np.mean(uu_vals))
    ss = float(np.mean(ss_vals))
    su = float(np.mean(su_vals))
    us = float(np.mean(us_vals))

    # Raw score in accuracy points. In [ -200, 200 ] when accuracies are percentages in [0, 100].
    raw = uu + ss - su - us

    # Normalization 1: scale to [-1, 1] under percentage accuracies.
    normalized_signed = raw / 200.0

    # Normalization 2: contrast ratio in [-1, 1], robust to global scaling.
    normalized_ratio = (uu + ss - su - us) / (uu + ss + su + us + eps)

    return {
        "disentanglement_raw": raw,
        "disentanglement_norm_signed": normalized_signed,
        "disentanglement_norm_ratio": float(normalized_ratio),
        "u_to_u_mean": uu,
        "s_to_s_mean": ss,
        "s_to_u_mean": su,
        "u_to_s_mean": us,
        "num_terms": float(len(uu_vals)),
    }



