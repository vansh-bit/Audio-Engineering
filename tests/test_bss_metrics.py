"""
Unit tests for BSS Evaluation Metrics: SI-SDR, SDR, and PIT matching.
"""

import numpy as np
import pytest
import torch

from evaluation.bss_metrics import calculate_sdr, calculate_sisdr, evaluate_separation_pit


@pytest.fixture
def synthetic_signals():
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig_a = (0.5 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
    sig_b = (0.5 * np.sin(2 * np.pi * 800.0 * t)).astype(np.float32)
    return sig_a, sig_b


def test_sisdr_identical_signal(synthetic_signals):
    sig_a, _ = synthetic_signals
    # An identical signal should have near-infinite SI-SDR (capped at 100 dB)
    score = calculate_sisdr(sig_a, sig_a)
    assert score >= 80.0


def test_sisdr_scale_invariance(synthetic_signals):
    sig_a, _ = synthetic_signals
    # SI-SDR must be strictly invariant to arbitrary scalar gain
    score_unit = calculate_sisdr(sig_a, sig_a)
    score_scaled = calculate_sisdr(sig_a, sig_a * 4.7)
    score_attenuated = calculate_sisdr(sig_a, sig_a * 0.12)

    assert abs(score_unit - score_scaled) < 1e-3
    assert abs(score_unit - score_attenuated) < 1e-3


def test_pit_permutation_resolution(synthetic_signals):
    sig_a, sig_b = synthetic_signals
    references = [sig_a, sig_b]

    # Inverted estimations: Output 0 is B, Output 1 is A
    estimations_inverted = [sig_b, sig_a]

    result = evaluate_separation_pit(references, estimations_inverted)
    # Optimal permutation should map ref 0 -> est 1, ref 1 -> est 0: (1, 0)
    assert result["optimal_permutation"] == [1, 0]
    assert result["mean_sisdr_db"] >= 80.0
    assert result["source_sisdr_db"][0] >= 80.0
    assert result["source_sisdr_db"][1] >= 80.0


def test_pit_delta_sisdr_improvement(synthetic_signals):
    sig_a, sig_b = synthetic_signals
    mixture = sig_a + sig_b

    # Simulated separation: add a small amount of residual bleed-through
    separated_a = sig_a + 0.05 * sig_b
    separated_b = sig_b + 0.05 * sig_a

    result = evaluate_separation_pit(
        references=[sig_a, sig_b],
        estimations=[separated_a, separated_b],
        mixture=mixture,
    )

    # Separation must show positive Delta SI-SDR improvement over the raw mixture
    assert result["mean_sisdr_improvement_delta_db"] > 5.0
    assert result["sisdr_improvement_delta_db"][0] > 5.0
    assert result["sisdr_improvement_delta_db"][1] > 5.0
