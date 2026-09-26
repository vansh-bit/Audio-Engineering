# Milestone M1 Completion & Verification Report

**Date**: September 25, 2026  
**Status**: COMPLETED & VERIFIED (100% Definition of Done achieved)  
**Milestone**: M1 — Controlled Data Pipeline  
**Execution Phase**: PHASE A (First Evaluation)  

---

## 1. Executive Summary
Milestone M1 has been successfully executed. We have constructed the **Controlled Synthetic Evaluation Suite (Track A)** required to evaluate Stage 1 Blind Source Separation (BSS) and calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR). Clean single-speaker reference audio stems representing authentic conversational Hindi and Hinglish (Hindi-English code-switched) speech have been curated, standardized, and parameterized into multi-speaker mixtures with exact time-aligned ground-truth references.

All 13 unit tests pass with zero errors, and all audio files and metadata manifests are fully operational.

---

## 2. Curated Single-Speaker Clean Reference Stems

All reference stems are standardized to uniform **16,000 Hz, 16-bit mono PCM `.wav`** format and saved in `data/controlled_eval_suite/clean_sources/`:

| Stem File Name | Speaker Gender / Linguistic Style | Duration | Peak Amp (dBFS) | RMS Energy (dB) | SNR Estimate (dB) | Ground Truth Transcript |
|---|---|---|---|---|---|---|
| `speaker_A_male_hinglish.wav` | Male / Code-Switched Hinglish | 4.715 s | -0.85 dB | -17.76 dB | 36.66 dB | *"उन्हें दस दिन तक rehab करना होगा और उसके बाद उनका fitness test लिया जाएगा"* |
| `speaker_B_male_hinglish.wav` | Male / Code-Switched Hinglish | 6.613 s | -0.63 dB | -20.01 dB | 47.01 dB | *"social network site facebook ने अब तक का सबसे बड़ा अधिग्रहण करते हुए whatsapp को खरीद लिया है"* |
| `speaker_C_female_indic.wav` | Female / Bilingual Indian English | 3.392 s | -3.01 dB | -17.92 dB | 43.20 dB | *"Author of the danger trail Philip Steels etc"* |
| `speaker_D_female_hindi.wav` | Female / Formal Indian Hindi | 6.000 s | -1.12 dB | -17.05 dB | 47.02 dB | *"यह एक वॉइस ऑडियो डेटासेट है जिसे आप AI प्रशिक्षण उद्देश्यों के लिए उपयोग कर सकते हैं..."* |

---

## 3. Generated Controlled Mixture Evaluation Suite

The mixture generator (`src/preprocessing/mixture_generator.py`) synthesized 6 distinct controlled acoustic conditions, outputting 18 audio files (6 mixtures + 12 time-aligned ground-truth references) in `data/controlled_eval_suite/synthetic_mixtures/`:

| Mixture ID | Speaker Pair | Overlap Target | Overlap Duration | SIR Target | Total Duration | Acoustic Condition / Scenario |
|---|---|---|---|---|---|---|
| `controlled_mix_01_pairAD_ov25_sir0` | Male A + Female D | 25% | 1.179 s | 0 dB | 9.536 s | Equal volume, natural turn interjection |
| `controlled_mix_02_pairAD_ov50_sir0` | Male A + Female D | 50% | 2.357 s | 0 dB | 8.357 s | Equal volume, heavy conversational overlap |
| `controlled_mix_03_pairAD_ov50_sir6` | Male A + Female D | 50% | 2.357 s | +6 dB | 8.357 s | Male dominant talker (+6 dB SIR) |
| `controlled_mix_04_pairBC_ov25_sir0` | Male B + Female C | 25% | 0.848 s | 0 dB | 9.157 s | Equal volume, natural turn interjection |
| `controlled_mix_05_pairBC_ov50_sir0` | Male B + Female C | 50% | 1.696 s | 0 dB | 8.309 s | Equal volume, conversational overlap |
| `controlled_mix_06_pairBC_ov50_sir_neg6` | Male B + Female C | 50% | 1.696 s | -6 dB | 8.309 s | Female dominant talker (-6 dB SIR) |

### Key Mathematical Property:
For every condition, the aligned ground-truth references $\mathbf{s}_{A,\text{ref}}$ and $\mathbf{s}_{B,\text{ref}}$ are zero-padded to the exact sample length of the mixture $\mathbf{x}$, satisfying the linear identity:
$$\mathbf{x}[n] = \mathbf{s}_{A,\text{ref}}[n] + \mathbf{s}_{B,\text{ref}}[n]$$
to within a floating-point tolerance of $< 10^{-4}$. This guarantees mathematical validity when computing SI-SDR in Milestone M2.

---

## 4. Metadata Manifest (`data/manifests/controlled_ground_truth.json`)
The manifest records complete acoustic and temporal metadata for every test pair:
* Exact start and end timestamps for each speaker.
* Paths to the original source stems and aligned ground-truth references.
* Target vs. actual overlap ratios and durations.
* Target Signal-to-Interference Ratio (SIR).
* Measured acoustic metrics (duration, peak dB, RMS energy, and estimated SNR).

---

## 5. Automated Unit Test Verification

The test suite was executed via `pytest tests/ -v`:

```
tests/test_audio_io.py::test_save_and_load_audio_roundtrip PASSED        [  7%]
tests/test_audio_io.py::test_audio_resampling PASSED                     [ 15%]
tests/test_audio_io.py::test_stereo_to_mono_conversion PASSED            [ 23%]
tests/test_audio_io.py::test_clipping_prevention PASSED                  [ 30%]
tests/test_audio_io.py::test_compute_audio_metrics PASSED                [ 38%]
tests/test_audio_io.py::test_mps_device_placement PASSED                 [ 46%]
tests/test_core.py::test_resolve_device PASSED                           [ 53%]
tests/test_core.py::test_base_module_lifecycle PASSED                    [ 61%]
tests/test_core.py::test_pipeline_state_serialization PASSED             [ 69%]
tests/test_mixture_generator.py::test_mixture_mathematical_reconstruction PASSED [ 76%]
tests/test_mixture_generator.py::test_overlap_durations PASSED           [ 84%]
tests/test_mixture_generator.py::test_sir_loudness_scaling PASSED        [ 92%]
tests/test_mixture_generator.py::test_generate_suite_and_manifest PASSED [100%]

============================== 13 passed in 2.18s ==============================
```

---

## 6. Milestone M1 Definition of Done Checklist

- [x] Controlled speech data: authentic Hindi/Hinglish speech curated across male and female speakers.
- [x] Audio format standardized to 16 kHz 16-bit mono PCM.
- [x] Parameterized mixture generator implemented (`src/preprocessing/mixture_generator.py`).
- [x] Overlap levels (25%, 50%) and SIR levels (0 dB, $\pm 6\text{ dB}$) verified.
- [x] Time-aligned ground truth reference stems generated with zero clipping distortion.
- [x] Master metadata manifest serialized to `data/manifests/controlled_ground_truth.json`.
- [x] 100% of unit tests pass cleanly (13/13).
- [x] Dataset generation fully reproducible via `PYTHONPATH=. python scripts/generate_controlled_suite.py`.

---

## 7. Next Step: Milestone M2
The project is now unblocked and ready for:
**Milestone M2: Stage 1 Source Separation Prototype**:
* Loading Meta's pre-trained Demucs (`htdemucs`) model.
* Running separation inference on `controlled_mix_01` through `controlled_mix_06`.
* Implementing Permutation Invariant Training (PIT) matching and SI-SDR metric calculation in `evaluation/bss_metrics.py`.
* Recording the first concrete quantitative result (+X dB SI-SDR improvement) and generating the isolated audio demonstration files.
