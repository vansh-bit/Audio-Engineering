"""
Real-World Audio Separation & Stress-Testing Demonstration Script.

Evaluates Conv-TasNet and Demucs on uncurated, in-the-wild audio from Track B:
- air_spontaneous_panel_01.wav (All India Radio 3-speaker debate with conversational cross-talk)
- vaani_rural_call_01.wav (Project Vaani rural phone-in call with cellular bandpass and line noise)
- air_formal_news_01.wav (Formal studio broadcast read-speech baseline)

Generates separated single-speaker audio files, visual spectrogram comparisons,
and a comprehensive benchmark report.
"""

import json
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from src.core.audio_io import compute_audio_metrics, load_audio, save_audio
from src.separation.conv_tasnet_wrapper import ConvTasNetSeparator
from src.separation.demucs_wrapper import DemucsSeparator


def generate_spectrogram_plot(
    mixture_path: Path,
    stem0_path: Path,
    stem1_path: Path,
    output_png: Path,
    title: str = "Real-World Radio Speech Separation (Conv-TasNet)",
):
    """Generates a 3-panel comparative spectrogram plot for real-world audio."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    tracks = [
        (mixture_path, "Raw Broadcast Mixture (Multi-Speaker Overlapping Cross-Talk)", "magma"),
        (stem0_path, "Separated Channel 0 (Isolated Primary Speaker)", "viridis"),
        (stem1_path, "Separated Channel 1 (Isolated Secondary Speaker / Cross-Talk)", "plasma"),
    ]

    for ax, (audio_p, subtitle, cmap) in zip(axes, tracks):
        data, sr = sf.read(str(audio_p))
        if data.ndim > 1:
            data = data[:, 0]
        
        Pxx, freqs, bins, im = ax.specgram(
            data,
            NFFT=512,
            Fs=sr,
            noverlap=384,
            cmap=cmap,
            vmin=-80,
            vmax=0,
        )
        ax.set_title(subtitle, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("Frequency (Hz)", fontsize=10)
        ax.set_ylim(0, 8000)
        fig.colorbar(im, ax=ax, format="%+2.0f dB", pad=0.02)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    fig.suptitle(title, fontsize=14, fontweight="heavy", y=0.98)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Spectrogram plot saved: {output_png}")


def main():
    print("=" * 80)
    print("STARTING REAL-WORLD AUDIO SEPARATION & STRESS-TEST DEMONSTRATION")
    print("=" * 80)

    input_dir = Path("data/real_world_eval_suite")
    output_base = Path("outputs/real_world_demo")
    manifest_path = Path("data/manifests/real_world_manifest.json")

    assert input_dir.exists(), f"Missing input directory: {input_dir}"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Initialize models on Apple Silicon Metal GPU
    device_pref = "auto"
    print(f"\nInitializing Conv-TasNet on device '{device_pref}'...")
    conv_tasnet = ConvTasNetSeparator(device=device_pref)
    conv_tasnet.initialize()
    print(f"Conv-TasNet initialized on: {conv_tasnet.device}")

    print(f"\nInitializing Demucs (htdemucs) on device '{device_pref}'...")
    demucs = DemucsSeparator(device=device_pref)
    demucs.initialize()
    print(f"Demucs initialized on: {demucs.device}")

    clips = manifest.get("clips", [])
    results_summary = []

    print("\nExecuting Real-World Separation Pipeline...")
    print("-" * 80)

    for clip in clips:
        clip_id = clip["clip_id"]
        file_path = Path(clip["file_path"])
        duration = clip["duration_seconds"]
        domain = clip["domain"]
        speech_style = clip["speech_style"]
        num_speakers = clip["num_speakers"]

        print(f"\nProcessing [{clip_id}] ({duration}s, Domain: {domain}, Speakers: {num_speakers})...")

        # 1. Run Conv-TasNet Separation
        tasnet_out_dir = output_base / "conv_tasnet" / clip_id
        t0 = time.perf_counter()
        stems_tasnet, meta_tasnet = conv_tasnet.separate_file(file_path, output_dir=tasnet_out_dir)
        tasnet_time = time.perf_counter() - t0
        tasnet_rtf = round(tasnet_time / duration, 3)

        spk0_path = Path(meta_tasnet["stems"][0]["file_path"])
        spk1_path = Path(meta_tasnet["stems"][1]["file_path"])

        metrics_spk0 = compute_audio_metrics(stems_tasnet[0], sample_rate=16000)
        metrics_spk1 = compute_audio_metrics(stems_tasnet[1], sample_rate=16000)

        print(f"  Conv-TasNet: RTF = {tasnet_rtf:.3f} ({tasnet_time:.2f}s inference)")
        print(f"    Channel 0 Peak: {metrics_spk0['peak_db']:.1f} dBFS | RMS: {metrics_spk0['rms_energy_db']:.1f} dBFS")
        print(f"    Channel 1 Peak: {metrics_spk1['peak_db']:.1f} dBFS | RMS: {metrics_spk1['rms_energy_db']:.1f} dBFS")

        # 2. Run Demucs Baseline Separation
        demucs_out_dir = output_base / "demucs" / clip_id
        t0 = time.perf_counter()
        stems_demucs, meta_demucs = demucs.separate_file(file_path, output_dir=demucs_out_dir)
        demucs_time = time.perf_counter() - t0
        demucs_rtf = round(demucs_time / duration, 3)
        print(f"  Demucs Baseline: RTF = {demucs_rtf:.3f} ({demucs_time:.2f}s inference)")

        # 3. Generate Spectrogram Plot for multi-speaker debate and rural call
        plot_path = output_base / f"{clip_id}_spectrogram_comparison.png"
        generate_spectrogram_plot(
            mixture_path=file_path,
            stem0_path=spk0_path,
            stem1_path=spk1_path,
            output_png=plot_path,
            title=f"Real-World Audio Separation: {clip_id} ({domain})",
        )

        clip_result = {
            "clip_id": clip_id,
            "domain": domain,
            "speech_style": speech_style,
            "duration_seconds": duration,
            "ground_truth_num_speakers": num_speakers,
            "conv_tasnet": {
                "inference_time_seconds": round(tasnet_time, 3),
                "rtf": tasnet_rtf,
                "separated_stems": [
                    {
                        "channel": 0,
                        "file_path": str(spk0_path),
                        "acoustic_metrics": metrics_spk0,
                    },
                    {
                        "channel": 1,
                        "file_path": str(spk1_path),
                        "acoustic_metrics": metrics_spk1,
                    },
                ],
            },
            "demucs": {
                "inference_time_seconds": round(demucs_time, 3),
                "rtf": demucs_rtf,
                "separated_stems": meta_demucs["stems"],
            },
            "spectrogram_plot": str(plot_path),
            "speaker_turns": clip.get("speaker_turns", []),
        }
        results_summary.append(clip_result)

    # Save Real-World Benchmark Results JSON
    results_json_path = output_base / "real_world_separation_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_name": "Real-World In-The-Wild Indian Audio Separation Stress-Test",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_clips_evaluated": len(results_summary),
                "device_used": str(conv_tasnet.device),
                "results": results_summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[✓] Real-World benchmark results saved to: {results_json_path}")
    print("=" * 80)
    print("REAL-WORLD STRESS-TEST DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
