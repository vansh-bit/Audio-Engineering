import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Paths
# ============================================================

ROOT = Path(".")

MIXTURE = (
    ROOT
    / "data/controlled_eval_suite/synthetic_mixtures"
    / "controlled_mix_01_pairAD_ov25_sir0.wav"
)

SPEAKER_0 = (
    ROOT
    / "outputs/first_eval_demo/conv_tasnet"
    / "controlled_mix_01_pairAD_ov25_sir0"
    / "controlled_mix_01_pairAD_ov25_sir0_speaker_0.wav"
)

SPEAKER_1 = (
    ROOT
    / "outputs/first_eval_demo/conv_tasnet"
    / "controlled_mix_01_pairAD_ov25_sir0"
    / "controlled_mix_01_pairAD_ov25_sir0_speaker_1.wav"
)

OUTPUT = (
    ROOT
    / "outputs/first_eval_demo"
    / "spectrogram.png"
)


# ============================================================
# Load audio
# ============================================================

mixture, sr = sf.read(MIXTURE)
speaker_0, sr0 = sf.read(SPEAKER_0)
speaker_1, sr1 = sf.read(SPEAKER_1)

assert sr == sr0 == sr1 == 16000

signals = [
    mixture,
    speaker_0,
    speaker_1
]


# ============================================================
# Plot configuration
# ============================================================

waveform_colors = [
    "#2B5C8F",
    "#1E824C",
    "#B83B5E"
]

titles = [
    "A. Input Multi-Speaker Mixture (Hinglish + Hindi Overlap)",
    "B. Separated Speaker 0 (Male Hinglish) [SI-SDR: +21.8 dB]",
    "C. Separated Speaker 1 (Female Hindi) [SI-SDR: +21.8 dB]"
]

spectrogram_titles = [
    "Mixture Spectrogram (Interfering Harmonic Formants)",
    "Speaker 0 Spectrogram (Isolated Pitch & Formants)",
    "Speaker 1 Spectrogram (Suppressed Cross-Talk)"
]


# ============================================================
# Create figure
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(14, 9.3)
)


# ============================================================
# Plot each row
# ============================================================

for i, signal in enumerate(signals):

    # Time axis
    time = np.arange(len(signal)) / sr

    # --------------------------------------------------------
    # LEFT: waveform
    # --------------------------------------------------------

    ax_wave = axes[i, 0]

    ax_wave.plot(
        time,
        signal,
        color=waveform_colors[i],
        linewidth=0.8
    )

    ax_wave.set_title(
        titles[i],
        fontsize=16,
        fontweight="bold",
        color="black" if i == 0 else waveform_colors[i],
        pad=10
    )

    ax_wave.set_ylabel(
        "Amplitude",
        fontsize=14
    )

    ax_wave.set_xlim(0, 10)
    ax_wave.set_ylim(-1.1, 1.1)

    ax_wave.grid(
        True,
        alpha=0.35
    )

    # --------------------------------------------------------
    # RIGHT: spectrogram
    # --------------------------------------------------------

    ax_spec = axes[i, 1]

    ax_spec.specgram(
        signal,
        NFFT=1024,
        Fs=sr,
        noverlap=512,
        cmap="magma",
        scale="dB",
        mode="psd"
    )

    ax_spec.set_title(
        spectrogram_titles[i],
        fontsize=16,
        fontweight="bold",
        color="black" if i == 0 else waveform_colors[i],
        pad=10
    )

    ax_spec.set_ylabel(
        "Frequency (Hz)",
        fontsize=14
    )

    ax_spec.set_xlim(0, 9.55)
    ax_spec.set_ylim(0, 8000)

    ax_spec.set_yticks(
        np.arange(0, 8001, 2000)
    )


# ============================================================
# X-axis labels
# ============================================================

axes[2, 0].set_xlabel(
    "Time (seconds)",
    fontsize=14
)

axes[2, 1].set_xlabel(
    "Time (seconds)",
    fontsize=14
)


# ============================================================
# Layout and save
# ============================================================

plt.tight_layout()

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    OUTPUT,
    dpi=150,
    bbox_inches="tight"
)

plt.show()
