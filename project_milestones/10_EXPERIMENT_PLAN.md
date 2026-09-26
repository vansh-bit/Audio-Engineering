# 10 — Detailed Scientific Experiment Plan & Matrix

## 1. Experimental Methodology & Rigorous Hygiene

To transition this project from a software integration exercise into a serious computational speech research project, every experiment follows a strict protocol:
1. **Hypothesis-Driven**: Every experiment tests a falsifiable technical hypothesis.
2. **Controlled Variables**: Only the independent variable under test is modified; random seeds, audio pre-processing, and hardware environments remain frozen.
3. **Dual Metric Tracking**: System accuracy (SI-SDR, DER, WER, ROUGE) and computational overhead (RTF, VRAM) are tracked concurrently.
4. **Reproducibility**: Experiment configurations are stored as versioned YAML files in `configs/experiments/`.

---

## 2. Core Experiment Matrix

### Experiment Set 1: Progressive Pipeline Architecture Ablation
* **Objective**: Quantify the downstream value added by each architectural stage.
* **Independent Variable**: Pipeline configuration depth (A, B, C, D).
* **Dependent Variables**: End-to-end WER (%), Speaker Attribution Accuracy (%), Output Usability Score.

| Experiment ID | Pipeline Configuration | Stage 1 (BSS) | Stage 2 (Diarization) | Stage 3 (ASR) | Stage 4 (LLM) | Core Hypothesis Tested |
|---|---|---|---|---|---|---|
| **EXP-1.1** | Pipeline A (Naive) | ❌ Bypassed | ❌ Bypassed | IndicASR | ❌ Bypassed | Severe ASR failure on overlapping multi-speaker audio. |
| **EXP-1.2** | Pipeline B (Separation Only) | ✅ Demucs | ❌ Bypassed | IndicASR | ❌ Bypassed | Separation eliminates acoustic collision, reducing word deletions. |
| **EXP-1.3** | Pipeline C (Traditional) | ✅ Demucs | ✅ Pyannote | IndicASR | ❌ Bypassed | Chronological speaker attribution achieved, but transcript disfluent. |
| **EXP-1.4** | Pipeline D (Full Proposed) | ✅ Demucs | ✅ Pyannote | IndicASR | ✅ Airavata | Superior semantic coherence, structured extraction, zero disfluency. |

---

### Experiment Set 2: Stage 1 Acoustic Difficulty Stress Tests
* **Objective**: Map the failure boundaries of Demucs source separation under adverse acoustic conditions.
* **Dataset**: Synthetic Controlled Mixture Suite (Track A).
* **Metrics**: Output SI-SDR (dB), SDR (dB), SIR (dB) using Permutation Invariant Training (PIT).

| Experiment ID | Overlap Ratio ($\Omega$) | Signal-to-Interference (SIR) | Background Noise (SNR) | Acoustic Scenario | Expected Outcome |
|---|---|---|---|---|---|
| **EXP-2.1** | $25\%$ | $0\text{ dB}$ (Equal loudness) | Clean ($\infty$) | Natural conversational turn-taking | High SI-SDR ($>+10\text{ dB}$) |
| **EXP-2.2** | $50\%$ | $0\text{ dB}$ | Clean ($\infty$) | Heated panel debate | Moderate SI-SDR ($+7\text{ to }+10\text{ dB}$) |
| **EXP-2.3** | $75\%$ | $0\text{ dB}$ | Clean ($\infty$) | Extreme simultaneous cross-talk | Degradation ($<+5\text{ dB}$) |
| **EXP-2.4** | $50\%$ | $+6\text{ dB}$ (Speaker A dominant) | Clean ($\infty$) | Dominant interviewer, quiet guest | Asymmetric separation quality |
| **EXP-2.5** | $50\%$ | $0\text{ dB}$ | $+15\text{ dB}$ (Moderate noise) | Real room reverberation | Phase distortion increase |
| **EXP-2.6** | $50\%$ | $0\text{ dB}$ | $+5\text{ dB}$ (Heavy field babble) | Agricultural mandi / street audio | Near-failure boundary; tests fallback |

---

### Experiment Set 3: Stage 2 Diarization & Separation Synergy
* **Objective**: Evaluate whether acoustic source separation improves diarization accuracy compared to diarizing raw mixed audio.
* **Metrics**: Diarization Error Rate (DER %), Speaker Confusion ($T_{\text{conf}}$ %), Jaccard Error Rate (JER %).

| Experiment ID | Input Audio Condition | Clustering Algorithm | Speaker Count ($K$) | Target Metric Focus |
|---|---|---|---|---|
| **EXP-3.1** | Raw Mixed Audio (No BSS) | Spectral Clustering | Ground Truth ($K=2$) | Baseline DER on overlapping speech |
| **EXP-3.2** | Separated Stems (With BSS) | Spectral Clustering | Ground Truth ($K=2$) | Measure reduction in Speaker Confusion ($T_{\text{conf}}$) |
| **EXP-3.3** | Separated Stems (With BSS) | Gaussian Mixture Model (GMM) | Ground Truth ($K=2$) | Spectral vs. GMM algorithmic trade-off |
| **EXP-3.4** | Separated Stems (With BSS) | Spectral Clustering | Estimated $K$ (Eigengap) | Robustness of automated speaker count estimation |

---

### Experiment Set 4: Stage 3 Indian Code-Switched ASR Benchmark
* **Objective**: Demonstrate why localized Indian speech architectures are essential compared to general-purpose Western baselines.
* **Dataset**: Code-switched Hinglish snippets from IndicVoices and Vaani.
* **Metrics**: Word Error Rate (WER %), Character Error Rate (CER %), Code-Switch Token Error Rate.

| Experiment ID | ASR Model Architecture | Target Language / Dialect | Preprocessing Condition | Target Hypothesis |
|---|---|---|---|---|
| **EXP-4.1** | OpenAI Whisper Baseline (`whisper-base`) | Hinglish (Hindi + English) | Raw unseparated audio | High WER ($>45\%$) due to code-switching flips and overlap |
| **EXP-4.2** | OpenAI Whisper Baseline (`whisper-base`) | Hinglish | Separated audio stems | Moderate WER reduction, but persistent code-switching errors |
| **EXP-4.3** | AI4Bharat `IndicWav2Vec` | Hinglish | Raw unseparated audio | Superior phonetic transcription, but overlap creates deletions |
| **EXP-4.4** | AI4Bharat `IndicWav2Vec` (Full System) | Hinglish | Separated audio stems | Lowest WER and CER; preserves code-switched terms |

---

### Experiment Set 5: Stage 4 Semantic Guardrailing & Hallucination Audit
* **Objective**: Measure the efficacy of strict negative prompt guardrails in eliminating factual hallucinations and preserving speaker attribution.
* **Metrics**: Hallucination Rate (%), Key Entity Recall (%), ROUGE-L, BERTScore.

| Experiment ID | Prompt Framework | LLM Backbone | Guardrail Constraints | Target Outcome |
|---|---|---|---|---|
| **EXP-5.1** | Standard Zero-Shot Prompt | Local 7B LLM | ❌ No negative constraints | High hallucination rate on noisy segments; invented numbers |
| **EXP-5.2** | Structured Few-Shot Prompt | Local 7B LLM | ✅ Factual Grounding Guardrail | Zero hallucination on crop prices; [Unintelligible] preserved |
| **EXP-5.3** | Schema-Constrained Prompt | Local 7B LLM | ✅ Grounding + Pydantic JSON Schema | $100\%$ schema validity, strict speaker attribution integrity |

---

## 3. Experiment Execution Workflow & Directory Layout

```
outputs/experiments/
├── exp_01_progressive_ablation/
│   ├── exp_01_results.json
│   └── ablation_wer_comparison.png
├── exp_02_bss_stress_tests/
│   ├── exp_02_results.csv
│   └── sisdr_vs_overlap_curve.png
├── exp_03_diarization_synergy/
│   ├── exp_03_results.json
│   └── der_confusion_matrix.png
├── exp_04_indic_asr_benchmark/
│   ├── exp_04_results.csv
│   └── whisper_vs_indic_wer.png
└── exp_05_llm_guardrails/
    ├── exp_05_audit_log.json
    └── hallucination_rate_bar.png
```

Each experiment script is independently executable via CLI flags:
```bash
# Example: Running Experiment 2 (BSS Stress Tests)
python -m evaluation.run_experiment --config configs/experiments/exp_02_bss.yaml
```
