# First Evaluation Checklist

> **CRITICAL GATE**: This checklist defines the exact boundary of **PHASE A (FIRST EVALUATION)**.  
> Every required item below must be satisfied and verified before the first evaluation presentation.

---

## Requirement 1 — Data
- [x] Dataset selected from approved open Indian corpora (IndicVoices, Nirantar, AIR-RS-DB)
- [x] Primary Hinglish (Hindi-English code-switched) focus established
- [x] Controlled mixture methodology defined (synthetic mixing with parameterized overlap and SIR)
- [x] Ground truth reference tracks and timestamps defined
- [x] Sample data prepared and validated (clean source stems and controlled mixture `.wav` files)

## Requirement 2 — Models
- [x] Meta's Demucs (`htdemucs`) selected and verified for Stage 1 Blind Source Separation
- [x] Backup/alternative architecture documented and implemented (Conv-TasNet)
- [x] Speaker diarization approach selected (`pyannote.audio` + Spectral Clustering / GMM)
- [x] Regional / code-switched ASR approach selected (`IndicWav2Vec` / `IndicASR`)
- [x] LLM post-processing approach selected (modular provider: Groq / OpenAI API during dev, Airavata local fallback)
- [x] Technical rationale for model choices documented

## Requirement 3 — Evaluation
- [x] Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) methodology implemented
- [x] Permutation Invariant Training (PIT) matching handled for output tracks
- [x] At least one concrete quantitative SI-SDR result measured against clean references (+15.45 dB improvement)
- [x] Future Diarization Error Rate (DER) methodology defined
- [x] Future Word Error Rate (WER) methodology defined with Indic text normalization
- [x] Future LLM factual hallucination and completeness evaluation defined
- [x] Computational metrics tracked (inference latency, processing time, RTF = 0.075)

## Requirement 4 — Demonstration
- [x] Real or controlled mixture processed through Stage 1 separation models
- [x] Separated single-speaker WAV files generated and saved in `outputs/first_eval_demo/`
- [x] Before/after audio comparison available (mixture audio vs. isolated speaker stems)
- [x] Short, self-contained 1-minute demo workflow prepared for presentation

## Requirement 5 — Flash Talk
- [x] 1–2 slide presentation prepared (`docs/flash_talk_slides.md`)
  - **Slide 1**: Problem statement, Indian speech acoustic challenges, and full system pipeline diagram
  - **Slide 2**: Selected data, selected models, evaluation metrics, concrete Stage 1 Demucs demo results, and next steps
- [x] 3–4 minute presentation narrative rehearsed and timed (`docs/flash_talk_script.md`)
- [x] Current implementation (Stage 1 BSS) vs. future milestones (Diarization, ASR, LLM) clearly demarcated

---

## FINAL GATE — PHASE A COMPLETE & SIGNED OFF

- **Gate Status**: **PASSED & SIGNED OFF**
- **Date**: 2026-09-25
- **Verified Deliverables**:
  1. Data pipeline & 6 controlled evaluation mixtures with ground-truth stems (`data/controlled_eval_suite/`)
  2. BSS model implementations (Demucs & Conv-TasNet on Apple Silicon MPS)
  3. SI-SDR & PIT mathematical evaluation with empirical +15.45 dB improvement
  4. Audio demonstration files (`outputs/first_eval_demo/`) and visual spectrograms (`spectrogram_comparison.png`)
  5. 2-slide presentation deck (`docs/flash_talk_slides.md`) and 3.5-minute presenter script (`docs/flash_talk_script.md`)
  6. Comprehensive test suite: 20/20 unit tests passing cleanly in ~6.2s
- **Next Phase**: Phase B — Post-Evaluation Downstream Execution (Milestones M4–M12).

