"""
High-Performance Audio I/O and Resampling Utility.

Uses soundfile and scipy.signal for rock-solid, C++-backed audio decoding and encoding
without fragile external dylib dependencies. Converts seamlessly to PyTorch tensors
with hardware device placement (MPS/CUDA/CPU).
"""

from math import log10
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import scipy.signal
import soundfile as sf
import torch


def load_audio(
    file_path: Union[str, Path],
    target_sr: Optional[int] = 16000,
    to_mono: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Loads an audio file, resamples to target_sr if specified and necessary, converts to mono,
    and returns a PyTorch tensor on the requested compute device.

    Args:
        file_path: Path to audio file.
        target_sr: Desired output sample rate (default 16000 Hz, None keeps original rate).
        to_mono: If True, multi-channel audio is averaged to single-channel mono.
        device: PyTorch device (mps/cuda/cpu) for tensor placement.

    Returns:
        Tuple of (waveform_tensor, sample_rate).
        Waveform shape is (channels, samples), e.g. (1, N).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {file_path}")

    # Read audio with soundfile in 32-bit floating point precision
    data, original_sr = sf.read(str(path), dtype="float32")

    # Ensure shape is (samples, channels)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    # Convert to mono if requested
    if to_mono and data.shape[1] > 1:
        data = np.mean(data, axis=1, keepdims=True)

    # Resample if sample rate mismatches and target_sr is specified
    output_sr = target_sr if target_sr is not None else original_sr
    if target_sr is not None and original_sr != target_sr:
        gcd = np.gcd(original_sr, target_sr)
        up = target_sr // gcd
        down = original_sr // gcd
        resampled_channels = []
        for ch in range(data.shape[1]):
            resampled = scipy.signal.resample_poly(data[:, ch], up, down)
            resampled_channels.append(resampled)
        data = np.column_stack(resampled_channels)

    # Transpose to standard PyTorch audio format: (channels, samples)
    data = data.T.astype(np.float32)
    tensor = torch.from_numpy(data)

    if device is not None:
        tensor = tensor.to(device)

    return tensor, output_sr



def save_audio(
    file_path: Union[str, Path],
    waveform: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 16000,
    subtype: str = "PCM_16",
) -> None:
    """
    Saves an audio waveform to disk as a standardized 16-bit PCM WAV.

    Args:
        file_path: Destination path.
        waveform: PyTorch tensor or NumPy array shaped (channels, samples) or (samples,).
        sample_rate: Audio sampling rate.
        subtype: WAV bit-depth encoding (default PCM_16).
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Move from GPU/MPS to CPU numpy
    if isinstance(waveform, torch.Tensor):
        data = waveform.detach().cpu().numpy()
    else:
        data = np.asarray(waveform)

    # Format to (samples, channels)
    if data.ndim == 2:
        data = data.T
    elif data.ndim == 1:
        data = data[:, np.newaxis]

    # Prevent digital clipping
    peak = np.max(np.abs(data))
    if peak > 1.0:
        data = data / peak

    sf.write(str(path), data, sample_rate, subtype=subtype)


def compute_audio_metrics(waveform: Union[torch.Tensor, np.ndarray], sample_rate: int = 16000) -> Dict[str, float]:
    """
    Computes acoustic features: duration, RMS energy (dB), peak amplitude, and SNR estimate.
    """
    if isinstance(waveform, torch.Tensor):
        data = waveform.detach().cpu().numpy().flatten()
    else:
        data = np.asarray(waveform).flatten()

    duration = float(len(data)) / float(sample_rate)
    peak = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
    peak_db = 20.0 * log10(max(peak, 1e-8))

    rms = float(np.sqrt(np.mean(data**2))) if len(data) > 0 else 0.0
    rms_db = 20.0 * log10(max(rms, 1e-8))

    # Basic noise-floor estimation from lowest 10th percentile energy frames
    frame_len = int(sample_rate * 0.025) # 25ms frames
    if len(data) >= frame_len:
        num_frames = len(data) // frame_len
        frames = data[: num_frames * frame_len].reshape(num_frames, frame_len)
        frame_energies = np.mean(frames**2, axis=1)
        noise_energy = float(np.percentile(frame_energies, 10))
        signal_energy = float(np.percentile(frame_energies, 90))
        snr_est = 10.0 * log10(max(signal_energy, 1e-8) / max(noise_energy, 1e-8))
    else:
        snr_est = 20.0

    return {
        "duration_seconds": round(duration, 3),
        "peak_amplitude": round(peak, 4),
        "peak_db": round(peak_db, 2),
        "rms_energy_db": round(rms_db, 2),
        "snr_estimate_db": round(snr_est, 2),
    }
