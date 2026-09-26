"""
Blind Source Separation Evaluation Metrics: SI-SDR, SDR, and PIT Matching.

Implements Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) with
Permutation Invariant Training (PIT) optimal assignment for multi-speaker
audio separation evaluation as defined in 04_STAGE_1_SOURCE_SEPARATION.md.
"""

from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


def calculate_sisdr(
    reference: Union[np.ndarray, torch.Tensor],
    estimation: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> float:
    """
    Computes Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in decibels.

    Args:
        reference: Ground-truth clean audio signal (1D array).
        estimation: Predicted / separated audio signal (1D array).
        eps: Small epsilon to prevent division by zero.
        zero_mean: If True, centers signals to zero-mean before projection.

    Returns:
        SI-SDR value in decibels (dB). Higher indicates superior separation fidelity.
    """
    if isinstance(reference, torch.Tensor):
        ref = reference.detach().cpu().numpy().flatten().astype(np.float64)
    else:
        ref = np.asarray(reference, dtype=np.float64).flatten()

    if isinstance(estimation, torch.Tensor):
        est = estimation.detach().cpu().numpy().flatten().astype(np.float64)
    else:
        est = np.asarray(estimation, dtype=np.float64).flatten()

    # Align lengths if minor sample mismatch
    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]

    if min_len == 0 or np.all(ref == 0) or np.all(est == 0):
        return -100.0

    if zero_mean:
        ref = ref - np.mean(ref)
        est = est - np.mean(est)

    # Orthogonal projection: s_target = (<est, ref> / ||ref||^2) * ref
    dot_product = np.dot(est, ref)
    ref_energy = np.dot(ref, ref)

    if ref_energy <= 1e-12:
        return -100.0

    alpha = dot_product / ref_energy
    s_target = alpha * ref
    e_noise = est - s_target

    target_energy = float(np.dot(s_target, s_target))
    noise_energy = float(np.dot(e_noise, e_noise))

    # If residual noise is at or below floating point precision threshold
    if target_energy > 0 and noise_energy <= 1e-13 * target_energy:
        return 100.0

    if noise_energy <= eps:
        return 100.0

    sisdr_val = 10.0 * np.log10(target_energy / (noise_energy + eps))
    return float(np.clip(sisdr_val, -100.0, 100.0))


def calculate_sdr(
    reference: Union[np.ndarray, torch.Tensor],
    estimation: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8,
) -> float:
    """
    Computes standard Signal-to-Distortion Ratio (SDR) in decibels.
    """
    if isinstance(reference, torch.Tensor):
        ref = reference.detach().cpu().numpy().flatten().astype(np.float64)
    else:
        ref = np.asarray(reference, dtype=np.float64).flatten()

    if isinstance(estimation, torch.Tensor):
        est = estimation.detach().cpu().numpy().flatten().astype(np.float64)
    else:
        est = np.asarray(estimation, dtype=np.float64).flatten()

    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]

    error = ref - est
    ref_pow = np.dot(ref, ref) + eps
    err_pow = np.dot(error, error) + eps

    sdr_val = 10.0 * np.log10(ref_pow / err_pow)
    return float(np.clip(sdr_val, -100.0, 100.0))


def evaluate_separation_pit(
    references: List[Union[np.ndarray, torch.Tensor]],
    estimations: List[Union[np.ndarray, torch.Tensor]],
    mixture: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """
    Evaluates multi-source separation using Permutation Invariant Training (PIT) matching.
    Identifies the optimal permutation assignment pi* that maximizes mean SI-SDR,
    resolving channel permutation ambiguity.

    Args:
        references: List of K ground-truth signals [s_1, s_2, ..., s_K].
        estimations: List of K predicted signals [s_hat_1, s_hat_2, ..., s_hat_K].
        mixture: Optional unseparated input mixture signal to compute baseline SI-SDR improvement.

    Returns:
        Dict with optimal permutation, per-source SI-SDR, mean SI-SDR, and Delta SI-SDR improvement.
    """
    K = len(references)
    if K != len(estimations):
        raise ValueError(f"Mismatch: {K} references vs. {len(estimations)} estimations.")

    best_perm = None
    best_mean_sisdr = -float("inf")
    best_scores: List[float] = []

    # Permutations of estimation indices: e.g. for K=2, (0, 1) and (1, 0)
    for perm in permutations(range(K)):
        perm_scores = []
        for ref_idx, est_idx in enumerate(perm):
            score = calculate_sisdr(references[ref_idx], estimations[est_idx])
            perm_scores.append(score)
        
        mean_score = float(np.mean(perm_scores))
        if mean_score > best_mean_sisdr:
            best_mean_sisdr = mean_score
            best_perm = perm
            best_scores = perm_scores

    assert best_perm is not None

    result: Dict[str, Any] = {
        "num_sources": K,
        "optimal_permutation": list(best_perm), # index mapping: reference i -> estimation perm[i]
        "mean_sisdr_db": round(best_mean_sisdr, 3),
        "source_sisdr_db": [round(s, 3) for s in best_scores],
    }

    # If mixture provided, compute Delta SI-SDR (separation improvement)
    if mixture is not None:
        mix_scores = []
        improvements = []
        for ref_idx in range(K):
            mix_score = calculate_sisdr(references[ref_idx], mixture)
            mix_scores.append(mix_score)
            delta = best_scores[ref_idx] - mix_score
            improvements.append(delta)

        result["baseline_mixture_sisdr_db"] = [round(s, 3) for s in mix_scores]
        result["mean_baseline_sisdr_db"] = round(float(np.mean(mix_scores)), 3)
        result["sisdr_improvement_delta_db"] = [round(d, 3) for d in improvements]
        result["mean_sisdr_improvement_delta_db"] = round(float(np.mean(improvements)), 3)

    return result
