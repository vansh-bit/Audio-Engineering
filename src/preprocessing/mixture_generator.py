"""
Controlled Mixture Generator for Synthetic Multi-Speaker Audio Evaluation.

Synthesizes overlapping multi-speaker acoustic mixtures with mathematically calibrated
overlap percentages (0% - 75%), Signal-to-Interference Ratios (SIR in dB), and optional
ambient noise. Generates time-aligned, zero-padded ground-truth reference stems
essential for Permutation Invariant Training (PIT) and SI-SDR evaluation in Stage 1.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from src.core.audio_io import compute_audio_metrics, load_audio, save_audio


class ControlledMixtureGenerator:
    """
    Synthesizes controlled overlapping audio mixtures from clean single-speaker stems.
    """

    def __init__(self, target_sr: int = 16000, random_seed: int = 42):
        self.target_sr = target_sr
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def mix_pair(
        self,
        stem_a_path: Union[str, Path],
        stem_b_path: Union[str, Path],
        overlap_ratio: float = 0.25,
        sir_db: float = 0.0,
        snr_noise_db: Optional[float] = None,
        output_dir: Optional[Union[str, Path]] = None,
        mixture_id: Optional[str] = None,
        speaker_a_id: str = "SPEAKER_00",
        speaker_b_id: str = "SPEAKER_01",
    ) -> Dict[str, Any]:
        """
        Mixes two clean audio stems with parameterized overlap and SIR.

        Args:
            stem_a_path: Path to clean speaker A audio file.
            stem_b_path: Path to clean speaker B audio file.
            overlap_ratio: Fraction of shorter speaker's duration that overlaps (0.0 to 1.0).
            sir_db: Desired Signal-to-Interference Ratio in decibels (0.0 = equal energy).
            snr_noise_db: Optional additive white noise SNR in decibels.
            output_dir: Directory to save generated WAVs and aligned references.
            mixture_id: Unique identifier for this mixture.
            speaker_a_id: Unique speaker identifier for A.
            speaker_b_id: Unique speaker identifier for B.

        Returns:
            Dict containing mixture metadata, timestamps, energy metrics, and file paths.
        """
        overlap_ratio = max(0.0, min(1.0, float(overlap_ratio)))
        
        # 1. Load clean stems standardized to 16 kHz mono
        tensor_a, _ = load_audio(stem_a_path, target_sr=self.target_sr, to_mono=True)
        tensor_b, _ = load_audio(stem_b_path, target_sr=self.target_sr, to_mono=True)

        sig_a = tensor_a.squeeze(0).cpu().numpy().astype(np.float32)
        sig_b = tensor_b.squeeze(0).cpu().numpy().astype(np.float32)

        len_a = len(sig_a)
        len_b = len(sig_b)

        # 2. Calibrate Signal-to-Interference Ratio (SIR)
        rms_a = float(np.sqrt(np.mean(sig_a**2))) if len_a > 0 else 1e-6
        rms_b = float(np.sqrt(np.mean(sig_b**2))) if len_b > 0 else 1e-6

        # Target ratio: rms_a / (scaled_rms_b) = 10^(sir_db / 20)
        scale_b = (rms_a / (rms_b * (10.0 ** (sir_db / 20.0)))) if rms_b > 0 else 1.0
        sig_b_scaled = (sig_b * scale_b).astype(np.float32)

        # 3. Calculate temporal offset based on overlap ratio
        overlap_samples = int(round(min(len_a, len_b) * overlap_ratio))
        start_b = max(0, len_a - overlap_samples)
        total_len = max(len_a, start_b + len_b)

        # 4. Construct time-aligned zero-padded ground truth reference signals
        aligned_a = np.zeros(total_len, dtype=np.float32)
        aligned_b = np.zeros(total_len, dtype=np.float32)

        aligned_a[:len_a] = sig_a
        aligned_b[start_b : start_b + len_b] = sig_b_scaled

        # 5. Acoustic superposition: x = s_A + s_B + n
        mixture = aligned_a + aligned_b

        if snr_noise_db is not None:
            mix_rms = float(np.sqrt(np.mean(mixture**2)))
            noise_rms = mix_rms / (10.0 ** (snr_noise_db / 20.0))
            noise = np.random.normal(0, noise_rms, total_len).astype(np.float32)
            mixture += noise

        # 6. Global peak normalization to prevent digital clipping
        peak = float(np.max(np.abs(mixture)))
        if peak > 0.95:
            norm_factor = 0.95 / peak
            mixture *= norm_factor
            aligned_a *= norm_factor
            aligned_b *= norm_factor

        # 7. Metadata calculation
        actual_overlap_sec = float(overlap_samples) / float(self.target_sr)
        duration_sec = float(total_len) / float(self.target_sr)
        start_a_sec = 0.0
        end_a_sec = float(len_a) / float(self.target_sr)
        start_b_sec = float(start_b) / float(self.target_sr)
        end_b_sec = float(start_b + len_b) / float(self.target_sr)

        if mixture_id is None:
            stem_a_name = Path(stem_a_path).stem
            stem_b_name = Path(stem_b_path).stem
            mixture_id = f"mix_{stem_a_name}_{stem_b_name}_ov{int(overlap_ratio*100)}_sir{int(sir_db)}"

        result_meta: Dict[str, Any] = {
            "mixture_id": mixture_id,
            "sample_rate_hz": self.target_sr,
            "total_duration_seconds": round(duration_sec, 3),
            "overlap_ratio_target": overlap_ratio,
            "overlap_duration_seconds": round(actual_overlap_sec, 3),
            "sir_target_db": float(sir_db),
            "snr_noise_db": snr_noise_db,
            "acoustic_metrics": compute_audio_metrics(mixture, sample_rate=self.target_sr),
            "sources": [
                {
                    "speaker_id": speaker_a_id,
                    "original_stem_path": str(stem_a_path),
                    "start_time_seconds": round(start_a_sec, 3),
                    "end_time_seconds": round(end_a_sec, 3),
                    "duration_seconds": round(end_a_sec - start_a_sec, 3),
                },
                {
                    "speaker_id": speaker_b_id,
                    "original_stem_path": str(stem_b_path),
                    "start_time_seconds": round(start_b_sec, 3),
                    "end_time_seconds": round(end_b_sec, 3),
                    "duration_seconds": round(end_b_sec - start_b_sec, 3),
                },
            ],
        }

        # 8. Save audio files if output_dir specified
        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            mix_file = out_path / f"{mixture_id}.wav"
            gt_a_file = out_path / f"{mixture_id}_gt_{speaker_a_id}.wav"
            gt_b_file = out_path / f"{mixture_id}_gt_{speaker_b_id}.wav"

            save_audio(mix_file, mixture, sample_rate=self.target_sr)
            save_audio(gt_a_file, aligned_a, sample_rate=self.target_sr)
            save_audio(gt_b_file, aligned_b, sample_rate=self.target_sr)

            result_meta["mixture_file_path"] = str(mix_file)
            result_meta["sources"][0]["aligned_ground_truth_path"] = str(gt_a_file)
            result_meta["sources"][1]["aligned_ground_truth_path"] = str(gt_b_file)

        return result_meta

    def generate_suite(
        self,
        stem_pairs: List[Tuple[Union[str, Path], Union[str, Path]]],
        overlap_ratios: List[float] = [0.25, 0.50],
        sir_db_values: List[float] = [0.0, 6.0],
        output_dir: Union[str, Path] = "data/controlled_eval_suite/synthetic_mixtures",
        manifest_path: Union[str, Path] = "data/manifests/controlled_ground_truth.json",
    ) -> List[Dict[str, Any]]:
        """
        Batch synthesizes a multi-condition controlled evaluation suite.
        """
        all_mixtures = []
        counter = 1

        for stem_a, stem_b in stem_pairs:
            for overlap in overlap_ratios:
                for sir in sir_db_values:
                    mix_id = f"controlled_mix_{counter:02d}_ov{int(overlap*100)}_sir{int(sir)}"
                    meta = self.mix_pair(
                        stem_a_path=stem_a,
                        stem_b_path=stem_b,
                        overlap_ratio=overlap,
                        sir_db=sir,
                        output_dir=output_dir,
                        mixture_id=mix_id,
                    )
                    all_mixtures.append(meta)
                    counter += 1

        # Save manifest JSON
        man_path = Path(manifest_path)
        man_path.parent.mkdir(parents=True, exist_ok=True)
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump({"total_mixtures": len(all_mixtures), "mixtures": all_mixtures}, f, indent=2)

        return all_mixtures

    def mix_multi_speaker(
        self,
        speakers: List[Dict[str, Any]],
        mixture_id: str,
        snr_noise_db: Optional[float] = None,
        output_dir: Optional[Union[str, Path]] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes an N-speaker mixture with custom temporal start offsets, gains,
        and aligned zero-padded ground-truth references for all speakers.

        Args:
            speakers: List of speaker specs:
                [
                    {"stem_path": "...", "speaker_id": "SPK_0", "start_offset_sec": 0.0, "gain_db": 0.0},
                    ...
                ]
            mixture_id: Unique identifier for this multi-speaker mixture.
            snr_noise_db: Optional additive white noise SNR in decibels.
            output_dir: Directory to save generated WAVs and aligned references.
            description: Semantic description of the acoustic scenario.

        Returns:
            Dict containing mixture metadata, timestamps, energy metrics, and file paths.
        """
        if not speakers:
            raise ValueError("speakers list cannot be empty")

        loaded_stems = []
        max_end_sample = 0

        for spk in speakers:
            stem_path = spk["stem_path"]
            spk_id = spk.get("speaker_id", f"SPEAKER_{len(loaded_stems):02d}")
            start_sec = max(0.0, float(spk.get("start_offset_sec", 0.0)))
            gain_db = float(spk.get("gain_db", 0.0))

            tensor, _ = load_audio(stem_path, target_sr=self.target_sr, to_mono=True)
            sig = tensor.squeeze(0).cpu().numpy().astype(np.float32)

            if gain_db != 0.0:
                sig = sig * (10.0 ** (gain_db / 20.0))

            start_sample = int(round(start_sec * self.target_sr))
            end_sample = start_sample + len(sig)
            if end_sample > max_end_sample:
                max_end_sample = end_sample

            loaded_stems.append({
                "speaker_id": spk_id,
                "stem_path": str(stem_path),
                "sig": sig,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_sec": start_sec,
                "end_sec": float(end_sample) / float(self.target_sr),
                "duration_sec": float(len(sig)) / float(self.target_sr),
            })

        # Construct time-aligned zero-padded ground-truth reference signals
        aligned_stems = []
        mixture = np.zeros(max_end_sample, dtype=np.float32)

        for item in loaded_stems:
            aligned = np.zeros(max_end_sample, dtype=np.float32)
            s_idx = item["start_sample"]
            e_idx = item["end_sample"]
            aligned[s_idx:e_idx] = item["sig"]
            aligned_stems.append(aligned)
            mixture += aligned

        if snr_noise_db is not None:
            mix_rms = float(np.sqrt(np.mean(mixture**2)))
            noise_rms = mix_rms / (10.0 ** (snr_noise_db / 20.0))
            noise = np.random.normal(0, noise_rms, max_end_sample).astype(np.float32)
            mixture += noise

        # Peak normalization to prevent clipping
        peak = float(np.max(np.abs(mixture)))
        if peak > 0.95:
            norm_factor = 0.95 / peak
            mixture *= norm_factor
            for i in range(len(aligned_stems)):
                aligned_stems[i] *= norm_factor

        total_duration_sec = float(max_end_sample) / float(self.target_sr)

        sources_meta = []
        for i, item in enumerate(loaded_stems):
            src_info = {
                "speaker_id": item["speaker_id"],
                "original_stem_path": item["stem_path"],
                "start_time_seconds": round(item["start_sec"], 3),
                "end_time_seconds": round(item["end_sec"], 3),
                "duration_seconds": round(item["duration_sec"], 3),
            }
            sources_meta.append(src_info)

        result_meta: Dict[str, Any] = {
            "mixture_id": mixture_id,
            "sample_rate_hz": self.target_sr,
            "total_duration_seconds": round(total_duration_sec, 3),
            "num_speakers": len(loaded_stems),
            "snr_noise_db": snr_noise_db,
            "description": description or f"{len(loaded_stems)}-speaker mixture",
            "acoustic_metrics": compute_audio_metrics(mixture, sample_rate=self.target_sr),
            "sources": sources_meta,
        }

        if output_dir is not None:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            mix_file = out_path / f"{mixture_id}.wav"
            save_audio(mix_file, mixture, sample_rate=self.target_sr)
            result_meta["mixture_file_path"] = str(mix_file)

            for i, item in enumerate(loaded_stems):
                spk_id = item["speaker_id"]
                gt_file = out_path / f"{mixture_id}_gt_{spk_id}.wav"
                save_audio(gt_file, aligned_stems[i], sample_rate=self.target_sr)
                result_meta["sources"][i]["aligned_ground_truth_path"] = str(gt_file)

        return result_meta

