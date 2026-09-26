"""
Audio Preprocessor Module for Indian Conversational Speech Engineering.

Provides robust acoustic preprocessing for downstream pipeline stages:
1. Audio validation (duration, sample rate, clipping detection, silence detection).
2. Loudness normalization (RMS/peak normalization to -20 dBFS / -1 dB peak).
3. Voice Activity Detection (Energy-based VAD with temporal hangover hysteresis).
4. Sliding-window segmentation for Stage 2 (Diarization) and Stage 3 (ASR).
5. Batch dataset ingestion and manifest generation.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from src.core.audio_io import compute_audio_metrics, load_audio, save_audio


@dataclass
class AudioValidationResult:
    """Detailed validation diagnostics for an audio file."""
    file_path: str
    is_valid: bool
    sample_rate: int
    duration_seconds: float
    channels: int
    peak_amplitude: float
    peak_db: float
    rms_energy_db: float
    is_clipped: bool
    is_silent: bool
    clipping_sample_count: int
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "is_valid": self.is_valid,
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 3),
            "channels": self.channels,
            "peak_amplitude": round(self.peak_amplitude, 4),
            "peak_db": round(self.peak_db, 2),
            "rms_energy_db": round(self.rms_energy_db, 2),
            "is_clipped": self.is_clipped,
            "is_silent": self.is_silent,
            "clipping_sample_count": self.clipping_sample_count,
            "issues": self.issues,
        }


@dataclass
class SpeechInterval:
    """Timestamped speech activity interval."""
    start_seconds: float
    end_seconds: float
    duration_seconds: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "duration_seconds": round(self.duration_seconds, 3),
        }


@dataclass
class WindowedSegment:
    """Sliding window audio segment with time boundaries."""
    segment_index: int
    start_seconds: float
    end_seconds: float
    waveform: torch.Tensor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "num_samples": self.waveform.shape[-1],
        }


class AudioPreprocessor:
    """
    Standardizes, validates, and segments conversational audio for multi-stage pipelines.
    """

    def __init__(
        self,
        target_sr: int = 16000,
        target_rms_db: float = -20.0,
        max_peak_amplitude: float = 0.90,
        vad_energy_threshold_db: float = -38.0,
        vad_frame_ms: float = 20.0,
        vad_hop_ms: float = 10.0,
        vad_min_speech_ms: float = 200.0,
        vad_min_silence_ms: float = 250.0,
    ):
        self.target_sr = target_sr
        self.target_rms_db = target_rms_db
        self.max_peak_amplitude = max_peak_amplitude
        self.vad_energy_threshold_db = vad_energy_threshold_db
        self.vad_frame_ms = vad_frame_ms
        self.vad_hop_ms = vad_hop_ms
        self.vad_min_speech_ms = vad_min_speech_ms
        self.vad_min_silence_ms = vad_min_silence_ms

    def validate_audio(
        self,
        file_path: Union[str, Path],
        min_duration_sec: float = 0.5,
        max_duration_sec: float = 600.0,
        clipping_threshold: float = 0.999,
        silence_rms_threshold_db: float = -60.0,
    ) -> AudioValidationResult:
        """
        Performs thorough acoustic sanity checks on an audio file.
        """
        path_obj = Path(file_path)
        issues: List[str] = []

        if not path_obj.exists():
            return AudioValidationResult(
                file_path=str(path_obj),
                is_valid=False,
                sample_rate=0,
                duration_seconds=0.0,
                channels=0,
                peak_amplitude=0.0,
                peak_db=-100.0,
                rms_energy_db=-100.0,
                is_clipped=False,
                is_silent=True,
                clipping_sample_count=0,
                issues=[f"File does not exist: {path_obj}"],
            )

        try:
            waveform, sr = load_audio(path_obj, target_sr=None, to_mono=False)
        except Exception as e:
            return AudioValidationResult(
                file_path=str(path_obj),
                is_valid=False,
                sample_rate=0,
                duration_seconds=0.0,
                channels=0,
                peak_amplitude=0.0,
                peak_db=-100.0,
                rms_energy_db=-100.0,
                is_clipped=False,
                is_silent=True,
                clipping_sample_count=0,
                issues=[f"Failed to load audio: {str(e)}"],
            )

        channels = waveform.shape[0]
        num_samples = waveform.shape[1]
        duration = float(num_samples) / float(sr)

        wav_np = waveform.cpu().numpy()
        peak_amp = float(np.max(np.abs(wav_np)))
        peak_db = 20.0 * np.log10(max(peak_amp, 1e-9))
        rms_val = float(np.sqrt(np.mean(wav_np**2)))
        rms_db = 20.0 * np.log10(max(rms_val, 1e-9))

        clipping_samples = int(np.sum(np.abs(wav_np) >= clipping_threshold))
        is_clipped = clipping_samples > 10
        is_silent = rms_db < silence_rms_threshold_db

        if duration < min_duration_sec:
            issues.append(f"Duration ({duration:.2f}s) is shorter than minimum ({min_duration_sec}s)")
        if duration > max_duration_sec:
            issues.append(f"Duration ({duration:.2f}s) exceeds maximum ({max_duration_sec}s)")
        if is_clipped:
            issues.append(f"Audio exhibits digital clipping: {clipping_samples} samples >= {clipping_threshold}")
        if is_silent:
            issues.append(f"Audio is silent: RMS energy {rms_db:.1f} dBFS < {silence_rms_threshold_db} dBFS")

        is_valid = len(issues) == 0

        return AudioValidationResult(
            file_path=str(path_obj),
            is_valid=is_valid,
            sample_rate=sr,
            duration_seconds=duration,
            channels=channels,
            peak_amplitude=peak_amp,
            peak_db=peak_db,
            rms_energy_db=rms_db,
            is_clipped=is_clipped,
            is_silent=is_silent,
            clipping_sample_count=clipping_samples,
            issues=issues,
        )

    def normalize_loudness(
        self,
        waveform: torch.Tensor,
        target_rms_db: Optional[float] = None,
        max_peak: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Applies gain scaling to match target RMS loudness while strictly enforcing peak bounds.
        """
        if target_rms_db is None:
            target_rms_db = self.target_rms_db
        if max_peak is None:
            max_peak = self.max_peak_amplitude

        wav = waveform.clone()
        rms_cur = float(torch.sqrt(torch.mean(wav**2)).item())

        if rms_cur < 1e-7:
            return wav  # Return silent tensor unchanged

        cur_rms_db = 20.0 * np.log10(rms_cur)
        gain_db = target_rms_db - cur_rms_db
        gain_linear = 10.0 ** (gain_db / 20.0)

        wav_scaled = wav * gain_linear

        # Peak limiter to prevent digital clipping
        current_peak = float(torch.max(torch.abs(wav_scaled)).item())
        if current_peak > max_peak:
            peak_reduction = max_peak / current_peak
            wav_scaled = wav_scaled * peak_reduction

        return wav_scaled

    def energy_vad(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        energy_threshold_db: Optional[float] = None,
    ) -> List[SpeechInterval]:
        """
        Detects active speech regions using short-time frame energy with temporal hangover.
        """
        if energy_threshold_db is None:
            energy_threshold_db = self.vad_energy_threshold_db

        # Ensure 1D numpy array
        if waveform.dim() > 1:
            sig = waveform.squeeze(0).cpu().numpy().astype(np.float32)
        else:
            sig = waveform.cpu().numpy().astype(np.float32)

        frame_len = int(round(sample_rate * (self.vad_frame_ms / 1000.0)))
        hop_len = int(round(sample_rate * (self.vad_hop_ms / 1000.0)))

        num_samples = len(sig)
        if num_samples < frame_len:
            # Short signal: check whole energy
            rms = float(np.sqrt(np.mean(sig**2))) if num_samples > 0 else 0.0
            db = 20.0 * np.log10(max(rms, 1e-9))
            if db >= energy_threshold_db:
                dur = float(num_samples) / float(sample_rate)
                return [SpeechInterval(0.0, dur, dur)]
            return []

        # Calculate frame energies
        num_frames = 1 + (num_samples - frame_len) // hop_len
        frame_energies_db = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            start = i * hop_len
            frame = sig[start : start + frame_len]
            rms = float(np.sqrt(np.mean(frame**2)))
            frame_energies_db[i] = 20.0 * np.log10(max(rms, 1e-9))

        # Initial binary decision
        is_speech = frame_energies_db >= energy_threshold_db

        # Morphological smoothing (hangover / hysteresis)
        min_speech_frames = max(1, int(round((self.vad_min_speech_ms / 1000.0) / (self.vad_hop_ms / 1000.0))))
        min_silence_frames = max(1, int(round((self.vad_min_silence_ms / 1000.0) / (self.vad_hop_ms / 1000.0))))

        # Bridge short silence gaps (hangover)
        smoothed = is_speech.copy()
        silence_run = 0
        for i in range(len(smoothed)):
            if not smoothed[i]:
                silence_run += 1
            else:
                if 0 < silence_run < min_silence_frames:
                    # Fill previous silence run with speech
                    smoothed[i - silence_run : i] = True
                silence_run = 0

        # Remove short speech bursts
        speech_run = 0
        for i in range(len(smoothed)):
            if smoothed[i]:
                speech_run += 1
            else:
                if 0 < speech_run < min_speech_frames:
                    smoothed[i - speech_run : i] = False
                speech_run = 0
        if 0 < speech_run < min_speech_frames:
            smoothed[len(smoothed) - speech_run : len(smoothed)] = False

        # Convert frame indices to time intervals
        intervals: List[SpeechInterval] = []
        in_segment = False
        start_frame = 0

        for i in range(len(smoothed)):
            if smoothed[i] and not in_segment:
                in_segment = True
                start_frame = i
            elif not smoothed[i] and in_segment:
                in_segment = False
                start_sec = (start_frame * hop_len) / sample_rate
                end_sec = min(float(num_samples) / sample_rate, ((i * hop_len) + frame_len) / sample_rate)
                dur = end_sec - start_sec
                if dur >= (self.vad_min_speech_ms / 1000.0):
                    intervals.append(SpeechInterval(start_sec, end_sec, dur))

        if in_segment:
            start_sec = (start_frame * hop_len) / sample_rate
            end_sec = float(num_samples) / sample_rate
            dur = end_sec - start_sec
            if dur >= (self.vad_min_speech_ms / 1000.0):
                intervals.append(SpeechInterval(start_sec, end_sec, dur))

        return intervals

    def sliding_window_segmentation(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        window_sec: float = 2.0,
        step_sec: float = 1.0,
    ) -> List[WindowedSegment]:
        """
        Slices continuous audio into uniform sliding windows for Stage 2 & 3 inference.
        """
        if waveform.dim() == 1:
            wav = waveform.unsqueeze(0)
        else:
            wav = waveform

        num_samples = wav.shape[-1]
        window_samples = int(round(window_sec * sample_rate))
        step_samples = int(round(step_sec * sample_rate))

        if num_samples <= window_samples:
            # Audio shorter than one window: pad to window length
            pad_len = window_samples - num_samples
            padded_wav = torch.nn.functional.pad(wav, (0, pad_len))
            return [
                WindowedSegment(
                    segment_index=0,
                    start_seconds=0.0,
                    end_seconds=float(num_samples) / float(sample_rate),
                    waveform=padded_wav,
                )
            ]

        segments: List[WindowedSegment] = []
        seg_idx = 0
        current_sample = 0

        while current_sample < num_samples:
            end_sample = min(current_sample + window_samples, num_samples)
            chunk = wav[:, current_sample:end_sample]

            # If last chunk is shorter than window_samples, zero-pad it
            if chunk.shape[-1] < window_samples:
                chunk = torch.nn.functional.pad(chunk, (0, window_samples - chunk.shape[-1]))

            start_t = float(current_sample) / float(sample_rate)
            end_t = float(end_sample) / float(sample_rate)

            segments.append(
                WindowedSegment(
                    segment_index=seg_idx,
                    start_seconds=start_t,
                    end_seconds=end_t,
                    waveform=chunk,
                )
            )

            current_sample += step_samples
            seg_idx += 1

            if end_sample >= num_samples:
                break

        return segments

    def preprocess_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Full end-to-end preprocessing of an audio file:
        Validates, resamples to 16 kHz mono, normalizes loudness, computes VAD intervals,
        and saves the standardized audio.
        """
        val_res = self.validate_audio(input_path)
        if not val_res.is_valid and val_res.is_silent:
            raise ValueError(f"Audio file is invalid or silent: {val_res.issues}")

        wav, _ = load_audio(input_path, target_sr=self.target_sr, to_mono=True)
        wav_norm = self.normalize_loudness(wav)

        vad_intervals = self.energy_vad(wav_norm, sample_rate=self.target_sr)
        metrics = compute_audio_metrics(wav_norm, sample_rate=self.target_sr)

        result: Dict[str, Any] = {
            "source_file": str(input_path),
            "target_sample_rate": self.target_sr,
            "validation": val_res.to_dict(),
            "preprocessed_metrics": metrics,
            "speech_intervals": [inv.to_dict() for inv in vad_intervals],
            "total_speech_duration_seconds": round(sum(inv.duration_seconds for inv in vad_intervals), 3),
        }

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            save_audio(out_p, wav_norm, sample_rate=self.target_sr)
            result["preprocessed_file"] = str(out_p)

        return result
