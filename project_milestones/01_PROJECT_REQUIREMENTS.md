# 01 — Project Requirements & Specification Analysis

## 1. Executive Summary & Document Authority
This document establishes the foundational technical specifications for the project **"Advanced Computational Speech Engineering: A Multi-Stage 'Who Spoke What and When' Audio Processing Pipeline"**.

The primary authority for the project requirements is the academic proposal document: `Audio_Engineering_AI_Project_Proposal.docx`. 
This specification decomposes the authoritative requirements, establishes an engineering framework, and explicitly segregates **mandatory academic requirements** from **our engineering enhancements** designed to ensure production quality, academic excellence, and high-impact resume defensibility.

### Formal Clarification: First Evaluation Scope
The professor has established the specific scope for the **First Evaluation Checkpoint**:
> *“We expect that by that time you will have figured out data, models and how to evaluate them, maybe with a demonstration. These will be taken as a flash talk (1-2 slides or a very short demo). Each team will get 3-4 minutes to give a talk, and we will evaluate your progress.”*

Accordingly, the project is structured into two sequential phases:
* **PHASE A — FIRST EVALUATION (M0 $\rightarrow$ M3)**: Focuses exclusively on answering:
  1. What data are we using? (`IndicVoices`, `Nirantar`, `AIR-RS-DB` with Hinglish focus).
  2. What models/approaches have we selected? (Demucs, pyannote, IndicASR, API-based LLM).
  3. How will we evaluate them? (Scale-Invariant SDR with PIT matching).
  4. Concrete working demonstration: Stage 1 Demucs source separation on controlled mixtures with measured SI-SDR.
  5. Flash talk package (1–2 slides, 3–4 minute timed narrative).
* **PHASE B — POST-EVALUATION / FULL PROJECT EXECUTION (M4 $\rightarrow$ M12)**: Downstream diarization, regional Hinglish ASR, guardrailed LLM structuring, end-to-end integration, error propagation analysis, benchmarking, and final thesis/portfolio defense.

---

## 2. Core Problem Statement & Acoustic Environment
In real-world speech processing, especially across the Indian subcontinent (e.g., community radio broadcasts, rural panel discussions, agricultural phone-in helplines, street-side debates), audio recordings are acoustic stress-tests. 

Standard commercial or open-source speech recognition engines (e.g., vanilla Whisper, Google Speech-to-Text) assume single-speaker turn-taking, studio acoustics, and standard monolingual syntax. Under real Indian field conditions, these engines degrade severely due to:
1. **Acoustic Overlap**: Multiple speakers interjecting, interrupting, and speaking concurrently.
2. **Linguistic Code-Switching**: Dynamic intra-sentential transitions between regional languages (Hindi, Bengali, Tamil, etc.) and Indian English (e.g., Hinglish, Benglish).
3. **Severe Environmental Degradation**: Low-cost mobile devices, uncalibrated electret microphones, clipping, non-stationary background noise (machinery, cattle, rural traffic, market babble).
4. **Spontaneous Speech Dynamics**: Hesitations, false starts, colloquial idioms, non-standard grammar, and high pitch/cadence variations across age and gender demographics.

The core technical objective is to engineer a resilient, modular, multi-stage processing system that ingests uncurated multi-speaker audio and deterministically outputs a structured, attributed, and semantically clean conversational report.

---

## 3. Authoritative Requirements vs. Engineering Enhancements

To maintain transparency with academic evaluation while pursuing elite engineering standards, all project components are categorized as follows:

| Category | Component / Milestone | Authoritative Source Requirement (Professor / Proposal) | Our Engineering Enhancement (Resume / Production Rigor) |
|---|---|---|---|
| **Front-End Audio** | Blind Source Separation (BSS) | Ingest multi-speaker radio broadcast `.wav`; separate overlapping vocal tracks into isolated single-speaker `.wav` channels using deep learning (Demucs or Conv-TasNet). | Synthetic controlled mixture generation pipeline; SI-SDR / SDR / SIR / SAR quantitative evaluation with Permutation Invariant Training (PIT) matching; noise/overlap stress tests. |
| **Speaker Attribution** | Acoustic Speaker Diarization | Extract speaker embeddings via `pyannote.audio`; apply unsupervised clustering (Spectral Clustering or GMM) to produce structured timeline JSON (`[start - end]: Speaker_X`). | Diarization Error Rate (DER) and Jaccard Error Rate (JER) evaluation on controlled multi-speaker timelines; silence thresholding; automated cluster count estimation ($k$ search via eigengap heuristic). |
| **Acoustic-to-Text** | Regional / Code-Switched ASR | Deploy localized Indian models (`IndicASR` or `IndicWav2Vec`); map regional phonemes to native scripts or standardized Latin representations for code-switched text (Hinglish/Benglish). | Comparative baseline evaluation against vanilla Whisper; Word Error Rate (WER) and Character Error Rate (CER) tracking; code-switch boundary preservation; confidence score logging. |
| **Semantic NLP** | Post-Processing & Structuring | Route raw transcripts through localized foundational LLMs (`Airavata` or `OpenHathi`); prompt engineering to fix syntactic errors from noise drops, extract key topics/market info, normalize code-switching, and output an executive summary with speaker dialogues. | Deterministic JSON schema enforcement; strict guardrailing against hallucinations; semantic consistency evaluation (ROUGE-L, BERTScore); quantitative comparison between raw vs. cleaned transcripts. |
| **System Integration** | Pipeline Software Artifact | Unified, documented Python implementation capable of processing raw multi-speaker audio end-to-end. | Decoupled architecture using abstract interfaces; configuration-driven pipelines (YAML); intermediate artifact caching; structured error handling; state serialization. |
| **Benchmarking** | System Benchmark Notebook | Validation notebook measuring computational latency and memory consumption of each module stage. | Standardized benchmark framework: Wall-clock latency, Processing Time, Real-Time Factor (RTF), Peak RAM usage, GPU VRAM allocation, and stage-by-stage throughput profiling across varying audio lengths. |
| **System Analysis** | Error Propagation Analysis | Mentioned as conceptual goal: "evaluating how error propagation cascades from early acoustic layers down to final text generations". | Formal mathematical and empirical error cascading study: quantifying how separation SI-SDR drops cascade into diarization DER degradation, which inflates ASR WER, which degrades LLM extraction fidelity. |

---

## 4. Pipeline Stages & Deliverable Specifications

### Stage 1: Blind Source Separation (BSS)
* **Goal**: Isolate overlapping speech waveforms into discrete single-speaker channels.
* **Input**: Single-channel or stereo mixed `.wav` audio (16 kHz, 16-bit PCM).
* **Target Architectures**: Meta's Hybrid Demucs (or Conv-TasNet).
* **Deliverable**: Discrete audio files (`speaker_1.wav`, `speaker_2.wav`, ..., `residual_noise.wav`).
* **Success Criteria**: Detectable acoustic separation without destructive phase cancellation or robotic musical artifacts.

### Stage 2: Acoustic Speaker Diarization
* **Goal**: Answer *"Who spoke when?"* chronologically without supervised speaker enrollment.
* **Input**: Separated speaker audio channels (and raw mixed audio for comparative baseline).
* **Target Architectures**: `pyannote.audio` embedding extractor combined with Spectral Clustering / Gaussian Mixture Models (GMMs).
* **Deliverable**: Standardized JSON timeline:
```json
[
  {
    "speaker_id": "SPEAKER_00",
    "start_time_seconds": 12.4,
    "end_time_seconds": 45.1,
    "confidence": 0.94
  },
  {
    "speaker_id": "SPEAKER_01",
    "start_time_seconds": 45.8,
    "end_time_seconds": 75.2,
    "confidence": 0.89
  }
]
```
* **Success Criteria**: Accurate turn-boundary detection with minimal speaker confusion and false alarm speech activity.

### Stage 3: Regional & Code-Switched ASR
* **Goal**: Answer *"What was spoken?"* across regional Indian dialects and code-switched Hindi-English (Hinglish).
* **Input**: Diarized audio segments aligned with speaker timestamps.
* **Target Architectures**: AI4Bharat's `IndicASR` / `IndicWav2Vec` (or fine-tuned Whisper-Indic variants where appropriate).
* **Deliverable**: Speaker-aligned transcript with native Devanagari script and standardized Latin representations for code-switched tokens:
```json
[
  {
    "speaker_id": "SPEAKER_00",
    "start_time": "00:12.400",
    "end_time": "00:18.200",
    "language": "hi-en",
    "raw_transcript": "आज market में गेहूं का price बहुत low है"
  }
]
```
* **Success Criteria**: High acoustic fidelity to phonetic output, preserving terminology without truncation on dialectal inflections.

### Stage 4: Semantic Post-Processing & LLM Guardrailing
* **Goal**: Synthesize, denoise, normalize, and extract structured intelligence from fragmented, noisy ASR output.
* **Input**: Chronological, speaker-attributed raw transcript.
* **Target Architectures**: Open-weight Indian foundational LLMs (`Airavata`, `OpenHathi`, or specialized instruction-tuned Indic-Llama).
* **Deliverable**: Production-grade JSON report and formatted Markdown document containing:
  1. Executive Summary (concise synthesis of discussion).
  2. Domain Keyword Tags (e.g., `#Agriculture`, `#MandiPrices`, `#WheatRate`).
  3. Speaker-Separated Cleaned Dialogue (preserving speaker identities and conversational intent while cleaning disfluencies).
  4. Extracted Action Items / Key Data Points (e.g., specific prices, locations, dates mentioned).
* **Success Criteria**: Zero hallucinated factual details; faithful preservation of speaker stance and intent; coherent multilingual synthesis.

---

## 5. Constraints, Non-Goals, and Guardrails

### Technical Constraints
1. **Compute Realism**: Must run reliably on standard workstation hardware (single modern GPU, e.g., Apple Silicon M-series via MPS / Metal, or NVIDIA T4/V100/A100 on Google Colab/local server).
2. **Reproducibility**: Environment specifications, random seeds, pinned model checkpoints, and deterministic processing flags must be standard.
3. **No Scratch Pre-training**: We will not train Demucs, Wav2Vec, or 7B parameter LLMs from scratch. We will leverage established pre-trained weights, evaluate their out-of-the-box domain robustness, and adapt inference pipelines.

### Non-Goals (What We Will NOT Build)
1. **No Real-Time Streaming Telephony Engine**: The system is designed for batch / file-based processing of recorded broadcasts, podcasts, and calls. Sub-second streaming audio buffering is out of scope.
2. **No Web / Mobile Frontend (Unless requested later)**: No React apps, Django portals, or complex mobile apps. The core software deliverables are a clean Python package/CLI and a Jupyter benchmark notebook.
3. **No Heavy Distributed Cloud Infrastructure**: No Kubernetes clusters, Kafka message queues, or distributed vector databases. Standard local filesystem caching and modular Python modules suffice.
