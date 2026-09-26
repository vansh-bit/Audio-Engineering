"""
Unit tests for ControlledMixtureGenerator and Ground Truth Alignment.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.core.audio_io import save_audio
from src.preprocessing.mixture_generator import ControlledMixtureGenerator


@pytest.fixture
def sample_stems(tmp_path: Path):
    sr = 16000
    # Stem A: 2.0s sine wave at 440 Hz
    t_a = np.linspace(0, 2.0, 32000, endpoint=False)
    sig_a = (0.5 * np.sin(2 * np.pi * 440.0 * t_a)).astype(np.float32)
    path_a = tmp_path / "stem_a.wav"
    save_audio(path_a, sig_a, sample_rate=sr)

    # Stem B: 3.0s sine wave at 880 Hz
    t_b = np.linspace(0, 3.0, 48000, endpoint=False)
    sig_b = (0.5 * np.sin(2 * np.pi * 880.0 * t_b)).astype(np.float32)
    path_b = tmp_path / "stem_b.wav"
    save_audio(path_b, sig_b, sample_rate=sr)

    return path_a, path_b


def test_mixture_mathematical_reconstruction(tmp_path: Path, sample_stems):
    path_a, path_b = sample_stems
    gen = ControlledMixtureGenerator(target_sr=16000)

    out_dir = tmp_path / "output_mix"
    meta = gen.mix_pair(
        stem_a_path=path_a,
        stem_b_path=path_b,
        overlap_ratio=0.50,
        sir_db=0.0,
        output_dir=out_dir,
        mixture_id="test_mix_reconstruct",
    )

    mix_file = Path(meta["mixture_file_path"])
    gt_a_file = Path(meta["sources"][0]["aligned_ground_truth_path"])
    gt_b_file = Path(meta["sources"][1]["aligned_ground_truth_path"])

    assert mix_file.exists()
    assert gt_a_file.exists()
    assert gt_b_file.exists()

    mix_data, _ = sf.read(str(mix_file))
    gt_a_data, _ = sf.read(str(gt_a_file))
    gt_b_data, _ = sf.read(str(gt_b_file))

    # All three files must have the EXACT same length
    assert len(mix_data) == len(gt_a_data) == len(gt_b_data)

    # Mathematical identity: mix = gt_a + gt_b (within 1e-4 float tolerance)
    diff = np.max(np.abs(mix_data - (gt_a_data + gt_b_data)))
    assert diff < 1e-4


def test_overlap_durations(sample_stems):
    path_a, path_b = sample_stems
    gen = ControlledMixtureGenerator(target_sr=16000)

    # Shorter stem is Stem A (2.0s = 32000 samples)
    # 25% overlap = 0.5s (8000 samples)
    meta_25 = gen.mix_pair(path_a, path_b, overlap_ratio=0.25)
    assert meta_25["overlap_duration_seconds"] == 0.5
    # Total duration = 2.0 + 3.0 - 0.5 = 4.5s
    assert abs(meta_25["total_duration_seconds"] - 4.5) < 0.01

    # 50% overlap = 1.0s (16000 samples)
    meta_50 = gen.mix_pair(path_a, path_b, overlap_ratio=0.50)
    assert meta_50["overlap_duration_seconds"] == 1.0
    # Total duration = 2.0 + 3.0 - 1.0 = 4.0s
    assert abs(meta_50["total_duration_seconds"] - 4.0) < 0.01


def test_sir_loudness_scaling(sample_stems):
    path_a, path_b = sample_stems
    gen = ControlledMixtureGenerator(target_sr=16000)

    # SIR = +6 dB: Speaker A should be louder than Speaker B
    meta = gen.mix_pair(path_a, path_b, overlap_ratio=0.25, sir_db=6.0)
    assert meta["sir_target_db"] == 6.0


def test_generate_suite_and_manifest(tmp_path: Path, sample_stems):
    path_a, path_b = sample_stems
    gen = ControlledMixtureGenerator(target_sr=16000)

    out_dir = tmp_path / "suite"
    manifest_path = tmp_path / "manifests" / "controlled_ground_truth.json"

    pairs = [(path_a, path_b)]
    suite = gen.generate_suite(
        stem_pairs=pairs,
        overlap_ratios=[0.25, 0.50],
        sir_db_values=[0.0, 6.0],
        output_dir=out_dir,
        manifest_path=manifest_path,
    )

    # 1 pair x 2 overlaps x 2 SIRs = 4 mixtures
    assert len(suite) == 4
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_mixtures"] == 4
    assert len(data["mixtures"]) == 4
    for mix in data["mixtures"]:
        assert Path(mix["mixture_file_path"]).exists()
        assert Path(mix["sources"][0]["aligned_ground_truth_path"]).exists()
        assert Path(mix["sources"][1]["aligned_ground_truth_path"]).exists()
