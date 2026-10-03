"""
Script to generate the official Controlled Evaluation Suite for Milestone M1.
"""

import json
from pathlib import Path
from src.preprocessing.mixture_generator import ControlledMixtureGenerator


def main():
    print("Initializing Controlled Evaluation Suite Generation...")
    clean_dir = Path("data/controlled_eval_suite/clean_sources")
    output_dir = Path("data/controlled_eval_suite/synthetic_mixtures")
    manifest_path = Path("data/manifests/controlled_ground_truth.json")

    stem_a = clean_dir / "speaker_A_female_hinglish.wav"
    stem_b = clean_dir / "speaker_B_male_hinglish.wav"
    stem_c = clean_dir / "speaker_C_female_indic.wav"
    stem_d = clean_dir / "speaker_D_male_hindi.wav"

    assert stem_a.exists(), f"Missing {stem_a}"
    assert stem_b.exists(), f"Missing {stem_b}"
    assert stem_c.exists(), f"Missing {stem_c}"
    assert stem_d.exists(), f"Missing {stem_d}"

    gen = ControlledMixtureGenerator(target_sr=16000, random_seed=42)

    # Define experimental conditions for Phase A First Evaluation
    conditions = [
        # Pair 1: Female Hinglish + Male Hindi
        {
            "stem_a": stem_a,
            "stem_b": stem_d,
            "overlap": 0.25,
            "sir": 0.0,
            "id": "controlled_mix_01_pairAD_ov25_sir0",
            "spk_a": "SPEAKER_FEMALE_HINGLISH",
            "spk_b": "SPEAKER_MALE_HINDI",
            "description": "Female Hinglish + Male Hindi at 25% overlap, equal volume (natural interjection)",
        },
        {
            "stem_a": stem_a,
            "stem_b": stem_d,
            "overlap": 0.50,
            "sir": 0.0,
            "id": "controlled_mix_02_pairAD_ov50_sir0",
            "spk_a": "SPEAKER_FEMALE_HINGLISH",
            "spk_b": "SPEAKER_MALE_HINDI",
            "description": "Female Hinglish + Male Hindi at 50% overlap, equal volume (conversational overlap)",
        },
        {
            "stem_a": stem_a,
            "stem_b": stem_d,
            "overlap": 0.50,
            "sir": 6.0,
            "id": "controlled_mix_03_pairAD_ov50_sir6",
            "spk_a": "SPEAKER_FEMALE_HINGLISH",
            "spk_b": "SPEAKER_MALE_HINDI",
            "description": "Female Hinglish + Male Hindi at 50% overlap, +6dB SIR (Female dominant talker)",
        },
        # Pair 2: Male Hinglish + Female Indic
        {
            "stem_a": stem_b,
            "stem_b": stem_c,
            "overlap": 0.25,
            "sir": 0.0,
            "id": "controlled_mix_04_pairBC_ov25_sir0",
            "spk_a": "SPEAKER_MALE_HINGLISH_2",
            "spk_b": "SPEAKER_FEMALE_INDIC",
            "description": "Male Hinglish 2 + Female Indic at 25% overlap, equal volume",
        },
        {
            "stem_a": stem_b,
            "stem_b": stem_c,
            "overlap": 0.50,
            "sir": 0.0,
            "id": "controlled_mix_05_pairBC_ov50_sir0",
            "spk_a": "SPEAKER_MALE_HINGLISH_2",
            "spk_b": "SPEAKER_FEMALE_INDIC",
            "description": "Male Hinglish 2 + Female Indic at 50% overlap, equal volume",
        },
        {
            "stem_a": stem_b,
            "stem_b": stem_c,
            "overlap": 0.50,
            "sir": -6.0,
            "id": "controlled_mix_06_pairBC_ov50_sir_neg6",
            "spk_a": "SPEAKER_MALE_HINGLISH_2",
            "spk_b": "SPEAKER_FEMALE_INDIC",
            "description": "Male Hinglish 2 + Female Indic at 50% overlap, -6dB SIR (Female dominant talker)",
        },
    ]

    all_mixtures = []
    print(f"Generating {len(conditions)} controlled evaluation mixtures...")

    for cond in conditions:
        meta = gen.mix_pair(
            stem_a_path=cond["stem_a"],
            stem_b_path=cond["stem_b"],
            overlap_ratio=cond["overlap"],
            sir_db=cond["sir"],
            output_dir=output_dir,
            mixture_id=cond["id"],
            speaker_a_id=cond["spk_a"],
            speaker_b_id=cond["spk_b"],
        )
        meta["description"] = cond["description"]
        all_mixtures.append(meta)
        print(f"  ✓ Generated: {cond['id']} (Duration: {meta['total_duration_seconds']}s, Overlap: {meta['overlap_duration_seconds']}s)")

    # Write master manifest
    manifest_data = {
        "dataset_name": "Indian Speech Controlled Mixture Evaluation Suite (Track A)",
        "sample_rate_hz": 16000,
        "format": "16-bit Mono PCM WAV",
        "total_mixtures": len(all_mixtures),
        "mixtures": all_mixtures,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\nMaster manifest saved successfully to: {manifest_path}")
    print(f"Total audio files in suite: {len(all_mixtures) * 3} (.wav files: 1 mix + 2 aligned ground truth stems per condition)")


if __name__ == "__main__":
    main()
