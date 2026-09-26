# 03 — Comprehensive Data Strategy & Corpus Analysis

## 1. Data Strategy Overview & Principles

In audio machine learning systems, data strategy dictates whether conclusions are scientifically valid or merely anecdotal. 

Our engineering data strategy is built on a **Dual-Evaluation Methodology**:
1. **Track A: Controlled Synthetic Evaluation Harness (Quantitative)**
   - Synthetically mixed clean single-speaker utterances with calibrated Signal-to-Noise Ratio (SNR), overlap percentages (0%, 25%, 50%, 75%), and loudness offsets.
   - Provides exact ground-truth waveforms and timestamps, enabling mathematical computation of **SI-SDR**, **DER**, and **WER**.
2. **Track B: In-the-Wild Real-World Indian Audio Corpus (Qualitative & Robustness)**
   - Uncurated acoustic cuts from regional Indian broadcasts and conversational archives.
   - Used to stress-test real-world performance on unmodeled microphone distortions, code-switching, and ambient babble.

---

## 2. In-Depth Corpus Analysis (Authoritative Corpora)

The proposal mandates the use of four open-source Indian speech resources. Each has distinct acoustic properties, annotations, and structural utility.

### Corpus 1: IndicVoices (AI4Bharat)
* **What it contains**: A massive foundational corpus comprising over 23,700 hours of natural, spontaneous, and extempore spoken audio across 22 scheduled Indian languages spanning 400+ districts. It captures spontaneous conversational speech alongside reading tasks.
* **Which Project Stage it Supports**:
  - **Stage 1 (Controlled Mixtures)**: Source for clean single-speaker utterances to generate synthetic overlapping multi-speaker mixtures.
  - **Stage 2 (Diarization Validation)**: Speaker-verified segments for validating speaker embedding clustering.
  - **Stage 3 (ASR Validation)**: High-quality reference transcriptions for calculating WER and CER across regional dialects.
* **Annotations & Metadata Available**: Speaker demographic metadata (gender, age group, native district/region), verified text transcriptions in native scripts, language IDs, and segment timestamps.
* **Intended Use**: Validation and Controlled Testing. Single-speaker slices will serve as pristine source material for controlled multi-speaker mixing.
* **Preprocessing Required**:
  - Audio resampled from native recording rates to uniform 16 kHz 16-bit mono PCM.
  - Segmenting long sessions into 3 to 10-second single-speaker chunks using metadata boundaries.
  - Energy normalization to -23 LUFS / peak -1.0 dB.
* **Limitations**: Audio quality varies across districts (field-recorded mobile devices); contains background environmental noise that must be filtered out when selecting pristine speaker references for Track A.

---

### Corpus 2: Project Vaani (IISc Bangalore & ARTPARK)
* **What it contains**: A large-scale open-source spoken corpus capturing linguistic diversity across 80+ districts in rural and urban India. Features real-world acoustic variations across multiple age groups, genders, educational backgrounds, and local dialects.
* **Which Project Stage it Supports**:
  - **Stage 3 (Regional & Code-Switched ASR)**: Evaluating ASR robustness on authentic rural accents and conversational cadence.
  - **Stage 4 (LLM Semantic Understanding)**: Raw conversational transcripts containing hyper-local agrarian terminology, local slang, and rural code-switching for LLM parsing tests.
* **Annotations & Metadata Available**: High-granularity district/taluk-level demographic metadata, age, gender, spontaneous prompt topics (e.g., agriculture, weather, local craft), and verified transcriptions.
* **Intended Use**: Testing and Real-World Domain Stress-Testing (especially for dialectal variability and agrarian vocabulary).
* **Preprocessing Required**:
  - Resampling to 16 kHz.
  - Removal of leading and trailing silence via Voice Activity Detection (VAD).
  - Normalization of regional script orthography.
* **Limitations**: Highly heterogeneous recording environments; many samples have low SNR or acoustic clipping. Not suitable for clean source separation ground truth, but ideal for ASR and LLM robustness benchmarking.

---

### Corpus 3: Nirantar (AI4Bharat)
* **What it contains**: Over 3,240 hours of human-transcribed, spontaneous conversational dialogue spanning 22 Indian languages and 208 distinct districts. Unlike read speech, Nirantar specifically documents authentic conversational turn-taking, colloquial hesitations, and real dialogue dynamics.
* **Which Project Stage it Supports**:
  - **Stage 2 (Diarization)**: Natural conversational turn-taking baseline (analyzing inter-speaker pause durations, natural interjections).
  - **Stage 3 (Conversational ASR)**: Validating ASR performance on spontaneous conversational speech (versus read speech).
  - **Stage 4 (LLM Summarization)**: Authentic two-party and multi-party transcripts to test LLM summarization, topic extraction, and speaker intent recovery.
* **Annotations & Metadata Available**: High-accuracy human-generated transcriptions, turn-level speaker demarcations, conversation context labels, and language identifiers.
* **Intended Use**: End-to-End Testing, Diarization validation, and LLM prompt testing.
* **Preprocessing Required**:
  - Aligning turn-level speaker tags with timestamped audio segments.
  - Filtering out non-speech audio events (laughter, coughs, microphone thumps).
* **Limitations**: High acoustic variability; multi-party overlap in raw recordings lacks isolated clean ground truth stems.

---

### Corpus 4: AIR-RS-DB (All India Radio Dataset)
* **What it contains**: A curated academic baseline containing pre-annotated speech segments from All India Radio (AIR) broadcasts. Crucially, it is structured into two explicit categories:
  1. *Formal Read Speech*: Studio-quality news readers with clean enunciation, high SNR, and minimal overlap.
  2. *Spontaneous Speech*: Panel guests, field interviews, agricultural phone-in programs, and debate shows featuring multiple speakers, overlapping dialogue, and varied acoustic quality.
* **Which Project Stage it Supports**:
  - **System Benchmark & Comparative Analysis**: Provides the exact empirical baseline to compare pipeline performance between "Formal Studio" vs. "Spontaneous Panel" conditions.
  - **Stage 1 (Separation Demonstration)**: Broadcast debate segments with real overlapping speech for qualitative BSS demonstration.
  - **Stage 2 (Diarization Demonstration)**: Multi-speaker panel discussions for real-world timeline generation.
* **Annotations & Metadata Available**: Formal vs. Spontaneous category tags, broadcast metadata, topic domains (agriculture, national news, regional updates), and segmented audio clips.
* **Intended Use**: Demonstration, Real-World Validation, and Comparative Ablation (Read vs. Spontaneous).
* **Preprocessing Required**:
  - Demuxing broadcast formats, splitting multi-minute programs into 30 to 120-second evaluation vignettes.
  - Manual verification of speaker count on test vignettes.
* **Limitations**: Limited formal word-level ground truth compared to IndicVoices; serves primarily as an acoustic benchmark rather than an ASR training set.

---

## 3. Practical Data Engineering & Storage Budget

We will **NOT** download terabytes of raw corpora onto local development machines. Instead, we establish a **Micro-Benchmark Dataset Protocol**:

```
data/
├── controlled_eval_suite/            # Track A: Synthetic evaluation suite (approx. 50 MB)
│   ├── clean_sources/                # 10 pristine single-speaker segments (IndicVoices/Nirantar)
│   │   ├── speaker_A_male_hi.wav
│   │   ├── speaker_B_female_hi.wav
│   │   └── ...
│   └── synthetic_mixtures/           # Pre-computed controlled mixtures
│       ├── mix_snr0_overlap25.wav
│       ├── mix_snr5_overlap50.wav
│       └── mix_snr_neg5_overlap75.wav
├── real_world_eval_suite/            # Track B: Real-world broadcast snippets (approx. 100 MB)
│   ├── air_spontaneous_panel_01.wav  # AIR radio discussion (multi-speaker, overlapping)
│   ├── air_formal_news_01.wav        # AIR studio broadcast (clean baseline)
│   └── vaani_rural_call_01.wav       # Vaani agricultural call (noisy, code-switched)
└── manifests/
    ├── controlled_ground_truth.json  # Exact start/end timestamps, source tracks, transcript
    └── real_world_annotations.json   # Approximate speaker turns and reference transcript
```

### Disk & Bandwidth Footprint
* Total target disk size: **< 200 MB**.
* Format: Uniform 16,000 Hz, 16-bit PCM mono `.wav`.
* Duration: 20–30 curated clips ranging from 15 seconds to 2 minutes each. This is statistically sufficient for reproducible metric evaluation while running rapidly in automated CI/CD and benchmarking notebooks.
