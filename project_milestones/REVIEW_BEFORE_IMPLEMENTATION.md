# REVIEW BEFORE IMPLEMENTATION: Project Architectural Baseline & Review Document

> **STATUS**: PROPOSAL FORMALLY REVIEWED & APPROVED BY USER.  
> **ROLE**: Senior AI/ML Engineer & Technical Project Architect.  
> **STATUS DIRECTIVE**: Implementation authorized to begin starting with Milestone M0 (Requirements, Scaffolding & Environment Architecture).

---

## 1. What the Project Is
The project is titled **"Advanced Computational Speech Engineering: A Multi-Stage 'Who Spoke What and When' Audio Processing Pipeline"**.

It addresses the fundamental acoustic challenge of real-world Indian conversational audio—such as community radio broadcasts, rural panel discussions, and agricultural phone-in programs. These environments exhibit severe acoustic degradation:
* Multiple concurrent speakers interrupting and talking over one another (**overlapping speech**).
* Rapid intra-sentential transitions between regional Indian languages and English (**code-switching** / Hinglish).
* Ambient environmental noise (machinery, rural background, market babble) recorded on low-cost microphones.
* Spontaneous speech characteristics (hesitations, false starts, dialectal colloquialisms).

Standard commercial and out-of-the-box ASR systems fail catastrophically under these conditions. This project engineers a resilient, end-to-end, multi-stage machine learning system that takes unconstrained raw audio and transforms it into a structured, attributed, and semantically clean conversational intelligence report.

---

## 2. What the Professor / Project Proposal Requires
From our rigorous analysis of the authoritative proposal (`Audio_Engineering_AI_Project_Proposal.docx`), the mandatory academic deliverables are:

1. **Four Core Pipeline Stages**:
   - **Stage 1: Blind Source Separation**: Implement deep learning architectures like Conv-TasNet or Meta's Demucs to isolate overlapping vocal tracks into single-speaker `.wav` files.
   - **Stage 2: Acoustic Speaker Diarization**: Extract speaker embeddings using `pyannote.audio` and cluster them using unsupervised ML (Spectral Clustering or GMM) to output a structured timeline JSON mapping time increments to unique speaker labels (`[00:12 - 00:45]: Speaker_A`).
   - **Stage 3: Regional & Code-Switched Transcription**: Deploy localized Indian speech models (`IndicASR` or `IndicWav2Vec`) mapping regional phonemes to native scripts or standardized Latin representations for code-switched text, outputting a timestamped speaker-attributed transcript.
   - **Stage 4: Semantic Post-Processing & Structuring (LLM Guardrailing)**: Route the raw transcript through a localized foundational LLM (`Airavata` or `OpenHathi`) with prompt engineering to clean syntactic noise drops, normalize code-switched phrasing, identify core topics, and generate an executive report with keyword tags and speaker dialogue.
2. **Core Software Artifacts**:
   - **Pipeline Source Code**: A unified, documented Python implementation capable of processing raw multi-speaker audio end-to-end to the final report.
   - **System Benchmark Notebook**: A validation notebook measuring computational latency and memory consumption of each module stage.
3. **Mandatory Dataset Resources**:
   - Utilize open-source Indian speech corpora: `IndicVoices`, `Project Vaani`, `Nirantar`, and `AIR-RS-DB`.

---

## 3. What We Plan to Build (Our Rigorous Engineering Extensions)
To elevate this project from a standard academic submission into a world-class, resume-defining portfolio piece, we have augmented the professor's requirements with production-level AI engineering standards:

* **Synthetic Controlled Mixture Evaluation Harness**: Because real broadcast recordings lack isolated ground-truth stems, we construct a calibrated synthetic mixture generator (with controlled overlap %, SIR, and SNR) enabling mathematical calculation of **SI-SDR**, **DER**, and **WER**.
* **Permutation Invariant Training (PIT) Metric Engine**: Proper handling of the channel permutation problem when evaluating source separation.
* **Progressive Pipeline Ablation Study**: Comparing four distinct pipeline depths (Raw $\rightarrow$ ASR vs. BSS $\rightarrow$ ASR vs. BSS $\rightarrow$ Diarize $\rightarrow$ ASR vs. Full System) to quantitatively prove the value of each stage.
* **Formal Cross-Stage Error Cascading Analysis**: Scientifically measuring how acoustic degradation in Stage 1 propagates through diarization clustering, inflates ASR error rates, and impacts LLM semantic fidelity.
* **Indic Unicode Normalization Layer**: Standardizing Devanagari script variations and matra encodings before computing error rates.
* **Strict LLM Negative Guardrails & Automated Factual Audit**: Prompt guardrails that eliminate hallucinations, backed by an automated script verifying that extracted prices, dates, and names exist in the source transcript.
* **Standardized Profiling Metrics**: Tracking Wall-Clock Latency, Processing Time, Real-Time Factor (RTF), Peak RAM, and Peak GPU VRAM across varying input lengths.
* **Decoupled OOP Software Architecture**: Clean abstract interfaces, Pydantic data contracts, reproducible YAML configurations, and isolated unit tests.

---

## 4. System Architecture
The system consists of five decoupled layers connected by strongly typed JSON contracts:
```
[Raw Audio .wav]
       │
       ▼
[0. Preprocessing & VAD] (16kHz Resampling, Peak Normalization, Noise Gating)
       │
       ▼
[1. Source Separation] (Meta Demucs U-Net -> Isolated Speaker Stems)
       │
       ▼
[2. Speaker Diarization] (pyannote.audio + Spectral Clustering + Eigengap k-estimation)
       │
       ▼
[3. Regional ASR] (Audio Segment Slicing + IndicWav2Vec / IndicASR -> Speaker Transcript)
       │
       ▼
[4. LLM Guardrails] (Airavata / OpenHathi + Negative Grounding -> Final Report JSON + MD)
```
* **State Management**: Encapsulated in an immutable `PipelineState` container passed across stages.
* **Resilience**: Energy thresholding prunes phantom tracks in single-speaker audio; low-confidence ASR tokens are flagged as `[Unintelligible]` rather than hallucinated by the LLM.

---

## 5. Dataset Strategy
We avoid downloading massive multi-terabyte datasets by instituting the **Micro-Benchmark Dataset Protocol** (< 200 MB disk footprint):
1. **Track A: Controlled Synthetic Evaluation Suite**
   - 10 clean single-speaker slices (5 male, 5 female, Hindi/Hinglish) from `IndicVoices` and `Nirantar`.
   - Parameterized mixtures synthesized across overlap ratios ($0\%, 25\%, 50\%, 75\%$), SIR ($0\text{ dB}, \pm 6\text{ dB}$), and additive noise ($+\infty, +15\text{ dB}, +5\text{ dB}$).
   - Serves as the quantitative benchmark for SI-SDR, DER, and WER.
2. **Track B: In-the-Wild Real-World Indian Audio Suite**
   - 3 multi-speaker panel snippets from `AIR-RS-DB` (Spontaneous Speech vs. Read Speech).
   - 1 field recording snippet from `Project Vaani` (agricultural call with ambient noise).
   - Serves as the qualitative robustness and domain demonstration suite.

---

## 6. Milestone Plan: Two-Phase Execution Architecture

```
================================================================================
FIRST EVALUATION COMPLETION POINT: MILESTONE 3
================================================================================
```

### PHASE A — FIRST EVALUATION (Immediate Focus)
* **M0: Project Scope, Data & Evaluation Design**: Problem scope (Hinglish focus), data selection (`IndicVoices`, `Nirantar`, `AIR-RS-DB`), model architecture selection, SI-SDR metric design, and core software scaffolding (**COMPLETED & VERIFIED**).
* **M1: Controlled Data Pipeline**: Slicing clean single-speaker references, implementing parameterized mixture synthesizer (25% & 50% overlap, 0 & +6 dB SIR), and generating ground-truth metadata manifests.
* **M2: Stage 1 — Source Separation Prototype**: Pre-trained Demucs (`htdemucs`) inference, generating separated single-speaker `.wav` files, and calculating quantitative SI-SDR with PIT matching.
* **M3: First Evaluation Package & Flash Talk**: Packaging the 4 required answers, audio demonstration files, 1–2 slide presentation, and 3–4 minute timed presenter script. Verified against [`FIRST_EVALUATION_CHECKLIST.md`](file:///Users/vanshsharma/Documents/AI%20Project/project_milestones/FIRST_EVALUATION_CHECKLIST.md).

### PHASE B — POST-EVALUATION / FULL PROJECT EXECUTION
* **M4: Full Data Pipeline & Corpus Expansion**: Expanding to `Project Vaani` and long-duration broadcast vignettes.
* **M5: Stage 2 — Speaker Diarization**: `pyannote.audio` embeddings + Spectral/GMM clustering + Eigengap $K$ estimation + DER evaluation.
* **M6: Stage 3 — Regional Hinglish ASR**: AI4Bharat `IndicWav2Vec` + Indic Unicode NFKC normalization + Whisper baseline comparator + WER/CER evaluation.
* **M7: Stage 4 — Semantic Post-Processing & LLM Guardrails**: Modular LLM provider layer (Groq/OpenAI APIs during dev, Airavata local fallback) + negative prompt guardrails + factual hallucination auditor.
* **M8: End-to-End System Integration & CLI Orchestrator**: Unified `src/pipeline.py` CLI orchestrator with intermediate artifact caching.
* **M9: Cross-Stage Error Propagation & Scientific Ablations**: Progressive pipeline ablations (A, B, C, D) and empirical error cascading study (SI-SDR vs. DER vs. WER).
* **M10: System Benchmarking & Computational Profiling Notebook**: `notebooks/01_system_benchmark.ipynb` profiling Latency, RTF, RAM, and GPU VRAM across varying audio durations.
* **M11: Real-World Indian Audio Validation**: Robustness stress-testing on uncurated radio debate recordings.
* **M12: Final Deliverables, Academic Report & Technical Interview Portfolio**: Final codebase packaging, comprehensive academic project report, and interview defense cheat-sheet.

---

## 7. Evaluation Strategy & Metrics
* **Stage 1 (Separation)**: Scale-Invariant Signal-to-Distortion Ratio (**SI-SDR** in dB), SDR, SIR evaluated via Permutation Invariant Training (PIT) matching.
* **Stage 2 (Diarization)**: Diarization Error Rate (**DER %** = Missed + False Alarm + Confusion) with $250\text{ ms}$ collar, and Jaccard Error Rate (**JER %**).
* **Stage 3 (ASR)**: Word Error Rate (**WER %**), Character Error Rate (**CER %**), Code-Switch Token Error Rate, with Unicode NFKC normalization.
* **Stage 4 (LLM)**: **Hallucination Rate %** (automated entity grounding check), ROUGE-L, BERTScore.
* **Cross-Cutting Benchmarks**: Wall-Clock Latency, Real-Time Factor (**RTF**), Peak RAM (MB), Peak GPU VRAM (MB).

---

## 8. Major Technical Decisions & Tradeoffs

1. **Demucs over Conv-TasNet for Stage 1**:
   - *Rationale*: Conv-TasNet suffers from metallic musical noise and phase artifacts on non-stationary background noise. Demucs's hybrid time-frequency U-Net preserves vocal formants necessary for downstream diarization embeddings.
2. **IndicWav2Vec over Vanilla Whisper for Stage 3**:
   - *Rationale*: Vanilla Whisper frequently flips languages, drops code-switched Hindi words, or loops on static. IndicWav2Vec is explicitly pre-trained on Indian linguistic phonology. Whisper is preserved strictly as a baseline comparator.
3. **Spectral Clustering over pure GMM for Stage 2**:
   - *Rationale*: Spectral clustering operates on non-linear manifold embeddings using cosine affinity, which outperforms GMM when cluster shapes are non-spherical, and enables automated $K$ estimation via the Eigengap heuristic.
4. **Offline Localized 7B LLM with Negative Guardrails for Stage 4**:
   - *Rationale*: Protects data privacy, allows fully reproducible local execution, and enables specialized Indian instruction models (Airavata).

---

## 9. Key Technical & Operational Risks

| Risk Category | Specific Risk Scenario | Concrete Mitigation Strategy |
|---|---|---|
| **Compute / Hardware** | GPU VRAM exhaustion on long audio clips during Demucs or pyannote passes. | Implement sliding-window chunking (chunk size 10s–30s with 2s cross-fade overlap) to strictly bound peak VRAM. |
| **Model Availability** | Hugging Face gated access or download delays for specific localized checkpoints. | Identify immediate mirror checkpoints and local fallback quantizations (GGUF / bitsandbytes). |
| **Acoustic Edge Cases** | Real radio broadcast contains 3+ simultaneous overlapping speakers where Demucs is configured for 2 sources. | Pipeline detects high residual energy in background noise stem and flags an unmodeled speaker alert. |
| **Indic Orthography** | WER artificially inflated due to alternate Devanagari Unicode codepoints (nuktas, conjuncts). | Mandatory pre-evaluation Unicode NFKC collation and punctuation stripping. |
| **LLM Hallucinations** | LLM invents crop rates or misattributes statements during summarization. | Negative prompt constraints + automated post-generation entity extraction matching against the raw transcript. |

---

## 10. Confirmed Technical Decisions (Reviewed & Approved by User)
All three key technical decisions have been formally finalized:

1. **Target Hardware & Execution Environment**:
   - **Local Autonomous Execution**: Autonomous local development and evaluation on Mac (Apple Silicon MPS / CPU) for rapid, automated iteration without manual bottlenecks.
   - **Google Colab Compatibility**: All pipeline modules and especially `notebooks/01_system_benchmark.ipynb` are architected device-agnostically (`cuda` / `mps` / `cpu`), featuring 1-click execution for Colab cloud GPUs (T4/A100).
2. **LLM Runtime Strategy (Modular Provider Architecture)**:
   - **Development Phase**: Use fast API endpoints (Groq / OpenAI / Sarvam AI) to accelerate development and focus engineering effort on prompt architecture, strict guardrails, JSON schema enforcement, and hallucination auditing.
   - **Modular Swappability**: Encapsulate LLM logic behind an abstract `BaseLLM` interface allowing seamless swapping between API providers (`GroqProvider`, `OpenAIProvider`, `SarvamProvider`) and local models (`LocalQuantizedProvider`).
   - **Final Milestone**: Local quantized 7B inference (Airavata / OpenHathi / Llama-3) is supported as an optional switchable backend, but is not a blocking dependency.
3. **Primary Linguistic Focus**:
   - **Hinglish (Hindi-English)** is confirmed as the primary controlled-experiment language. The proposal explicitly highlights Hinglish as the target code-switched paradigm, while the pipeline abstractions remain fully extensible to broader Indian languages (Bengali, Tamil, etc.).

---

## 11. Things That Should NOT Be Implemented Yet
To prevent distraction and scope creep, the following components are **strictly out of scope** at this stage:
- ❌ Do NOT download multi-gigabyte or terabyte raw speech corpora.
- ❌ Do NOT train neural network weights from scratch.
- ❌ Do NOT build web user interfaces, mobile apps, or frontend dashboards.
- ❌ Do NOT deploy complex cloud services, Kubernetes clusters, or microservice queues.
- ❌ Do NOT generate fabricated performance numbers or commit unverified code.

---

## 12. Recommended Next Step After Approval
Once you have reviewed this document and given formal approval:
1. We will begin with **Milestone M0: Requirements, Scaffolding & Environment Architecture**.
2. We will set up the project folder structure (`src/`, `configs/`, `data/`, `tests/`, `evaluation/`, `notebooks/`), initialize the virtual environment, pin dependencies, and verify PyTorch audio acceleration on your local machine.
3. Every milestone will be executed one at a time, strictly verified against its **Definition of Done (DoD)** before advancing.

---

```
================================================================================
DO NOT START IMPLEMENTATION UNTIL THIS PLAN HAS BEEN REVIEWED.
================================================================================
```
