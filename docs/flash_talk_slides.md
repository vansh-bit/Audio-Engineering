# Flash Talk Presentation (1–2 Slides)
**Project Title**: Advanced Computational Speech Engineering: "Who Spoke What and When"  
**Academic Checkpoint**: First Evaluation Presentation (3–4 Minutes)  

---

## SLIDE 1: Problem Statement & End-to-End System Architecture

### 1. The Real-World Challenge: Indian Conversational Acoustics
* **The Acoustic Reality**: Real-world Indian broadcasts, panel discussions, and agrarian call-ins are chaotic acoustic stress-tests:
  - **Severe Overlapping Speech**: Multiple speakers frequently interject and talk over one another.
  - **Dynamic Code-Switching**: Rapid transitions between regional languages and English (**Hinglish**).
  - **Uncurated Acoustics**: Low-cost mobile microphones, ambient village/market noise, and reverberation.
* **The Core Objective**: Engineer a resilient multi-stage pipeline answering:
  > **"WHO spoke WHAT and WHEN?"**
* **Primary Controlled Linguistic Target**: **Hindi-English (Hinglish)** code-switching.

---

### 2. Multi-Stage Pipeline Architecture & Development Status

```
┌───────────────────────────────────────────────────────────────────────────┐
│                      CURRENTLY IMPLEMENTED & VERIFIED                     │
├───────────────────────────────────────────────────────────────────────────┤
│  [Raw Multi-Speaker Audio]                                                │
│         │                                                                 │
│         ▼                                                                 │
│  STAGE 1: BLIND SOURCE SEPARATION (BSS)                                   │
│  • Primary Model: Conv-TasNet (Time-domain TCN separation, Libri2Mix)    │
│  • Alternative Evaluated: Meta's Hybrid Demucs (htdemucs)                │
│  • Output: Isolated Single-Speaker .wav Channels                          │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────────────────────────────────────────────┐
│                    PHASE B: PLANNED ROADMAP MILESTONES                    │
├───────────────────────────────────────────────────────────────────────────┤
│         ▼                                                                 │
│  STAGE 2: ACOUSTIC SPEAKER DIARIZATION                                    │
│  • Architecture: pyannote.audio Neural Embeddings + Spectral Clustering   │
│  • Evaluation: Diarization Error Rate (DER) with NIST collar tolerance    │
│  • Output: Chronological Speaker Timeline JSON [00:12 - 00:45]: Speaker_A │
│         │                                                                 │
│         ▼                                                                 │
│  STAGE 3: REGIONAL & CODE-SWITCHED ASR                                    │
│  • Architecture: AI4Bharat IndicWav2Vec / IndicASR (vs. Whisper Baseline) │
│  • Evaluation: Word Error Rate (WER) with Indic Unicode Normalization     │
│  • Output: Speaker-Attributed Bilingual Transcript                        │
│         │                                                                 │
│         ▼                                                                 │
│  STAGE 4: SEMANTIC POST-PROCESSING & LLM GUARDRAILING                     │
│  • Architecture: Modular LLM Layer (Airavata / Local 7B / Fast API)      │
│  • Evaluation: Factual Hallucination Audit (0% invented numbers/prices)   │
│  • Output: Executive Summary, Topic Tags & Cleaned Dialogue Report        │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## SLIDE 2: Data Strategy, Evaluation & Stage 1 Working Demonstration

### 1. Data Strategy: Controlled Synthetic Evaluation Harness
* **The Challenge**: Real radio broadcasts lack isolated ground-truth audio stems for each speaker.
* **Our Solution**: Extracted real human conversational speech from open Indian corpora (`IndicVoices`, `Nirantar`, `AIR-RS-DB`) and synthesized calibrated controlled mixtures:
  - Overlap Ratios: $25\%$ (natural interjection) and $50\%$ (conversational cross-talk).
  - Relative Loudness: $\text{SIR} = 0\text{ dB}$ (equal volume) and $\pm 6\text{ dB}$ (dominant talker).
  - Aligned Ground Truth: Zero-padded clean stems satisfying $\mathbf{x} = \mathbf{s}_A + \mathbf{s}_B$ for mathematical metric evaluation.

---

### 2. Evaluation Methodology: Scale-Invariant SDR (SI-SDR) & PIT
* **Metric**: SI-SDR measures true waveform separation fidelity invariant to overall gain scaling:
  $$\text{SI-SDR} = 10 \log_{10} \left( \frac{\|\mathbf{s}_{\text{target}}\|^2}{\|\mathbf{e}_{\text{noise}}\|^2} \right)$$
* **Permutation Invariant Training (PIT)**: Automatically evaluates all $K!$ output channel permutations ($\pi^*$) to prevent false penalties from channel reordering.

---

### 3. Empirical Results: Conv-TasNet vs. Demucs Comparison
Evaluated on Apple Silicon Metal GPU (`mps:0`):

| Metric / Dimension | Baseline Mixture | Meta's Demucs (`htdemucs`) | Conv-TasNet (`conv_tasnet`) |
|---|---|---|---|
| **Separation SI-SDR (Mean)** | -0.003 dB | -8.598 dB | **+15.448 dB (Superior)** |
| **SI-SDR Improvement ($\Delta$)** | 0.00 dB | -8.59 dB | **+15.451 dB Gain** |
| **Real-Time Factor (RTF)** | 1.000 | 0.539 | **0.075 (13x Faster than Real Time)** |
| **Architectural Role** | Unseparated | Vocal/Speech vs. Background Denoiser | **Isolated Multi-Speaker Separation** |

---

### 4. Working Demonstration (Audio & Spectrogram Proof)
* **Sample**: `controlled_mix_01` (9.5s, 25% overlap, Female Hinglish + Male Hindi).
* **Demonstration Audio Artifacts**:
  1. *Original Mixture*: Overlapping concurrent speech where neither speaker is cleanly separable.
  2. *Separated Speaker 0 (Female Hinglish)*: **21.8 dB SI-SDR**, SNR **61.9 dB** (clean enunciation).
  3. *Separated Speaker 1 (Male Hindi)*: **21.8 dB SI-SDR**, SNR **67.8 dB** (cross-talk eliminated).
* **Visual Verification**: Spectrogram analysis (`outputs/first_eval_demo/spectrogram_comparison.png`) confirms complete elimination of cross-speaker formant interference.

---

### 5. Next Steps (Phase B Execution)
* **Milestone M5**: Diarize separated stems using `pyannote.audio` and Spectral Clustering (DER).
* **Milestone M6**: Transcribe isolated channels using AI4Bharat `IndicWav2Vec` (WER).
* **Milestone M7**: Implement LLM guardrails for factual synthesis and structured executive reporting.
