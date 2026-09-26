"""
Unit tests for Audio I/O, Resampling, Metrics, and Normalization.
"""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch

from src.core.audio_io import compute_audio_metrics, load_audio, save_audio


def test_save_and_load_audio_roundtrip(tmp_path: Path):
    sr = 16000
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    audio_file = tmp_path / "test_roundtrip.wav"
    save_audio(audio_file, sine, sample_rate=sr)
    assert audio_file.exists()

    tensor, loaded_sr = load_audio(audio_file, target_sr=sr)
    assert loaded_sr == 16000
    assert tensor.shape == (1, int(sr * duration))
    assert tensor.dtype == torch.float32


def test_audio_resampling(tmp_path: Path):
    # Create 44.1 kHz audio file
    original_sr = 44100
    target_sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(original_sr * duration), endpoint=False)
    sine = (0.3 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

    audio_file = tmp_path / "test_44k.wav"
    sf.write(str(audio_file), sine, original_sr)

    # Load with automatic resampling to 16 kHz
    tensor, loaded_sr = load_audio(audio_file, target_sr=target_sr)
    assert loaded_sr == 16000
    # Expected samples should be within 1-2 samples of 16000 due to polyphase filter boundary
    assert abs(tensor.shape[1] - 16000) <= 2


def test_stereo_to_mono_conversion(tmp_path: Path):
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = 0.4 * np.sin(2 * np.pi * 300.0 * t)
    right = 0.6 * np.sin(2 * np.pi * 600.0 * t)
    stereo = np.column_stack([left, right]).astype(np.float32)

    audio_file = tmp_path / "test_stereo.wav"
    sf.write(str(audio_file), stereo, sr)

    tensor, loaded_sr = load_audio(audio_file, target_sr=sr, to_mono=True)
    assert tensor.shape[0] == 1 # Mono
    assert tensor.shape[1] == int(sr * duration)


def test_clipping_prevention(tmp_path: Path):
    sr = 16000
    # Over-amplitude signal (> 1.0)
    over_amp = np.array([2.5, -3.0, 1.8, -0.5], dtype=np.float32)
    audio_file = tmp_path / "test_clip.wav"

    save_audio(audio_file, over_amp, sample_rate=sr)
    data, _ = sf.read(str(audio_file))
    assert np.max(np.abs(data)) <= 1.0


def test_compute_audio_metrics():
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    metrics = compute_audio_metrics(sine, sample_rate=sr)
    assert metrics["duration_seconds"] == 2.0
    assert 0.49 <= metrics["peak_amplitude"] <= 0.51
    assert -7.0 <= metrics["peak_db"] <= -5.0
    assert "rms_energy_db" in metrics
    assert "snr_estimate_db" in metrics


def test_mps_device_placement(tmp_path: Path):
    if torch.backends.mps.is_available():
        sr = 16000
        sine = np.zeros(16000, dtype=np.float32)
        audio_file = tmp_path / "test_mps.wav"
        save_audio(audio_file, sine, sample_rate=sr)

        device = torch.device("mps")
        tensor, _ = load_audio(audio_file, target_sr=sr, device=device)
        assert tensor.device.type == "mps"
