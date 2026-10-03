"""
Builds a single-file, minimalist, dark-mode HTML5 Audio Showcase Player (demo.html).
Embeds audio waveforms as base64 data URIs for 100% offline, zero-CORS instant playback.
"""

import base64
from pathlib import Path

def encode_audio_base64(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"

def main():
    root = Path(__file__).resolve().parent.parent

    # Track A files
    track_a_mix = root / "data/controlled_eval_suite/synthetic_mixtures/controlled_mix_01_pairAD_ov25_sir0.wav"
    track_a_spk0 = root / "outputs/first_eval_demo/conv_tasnet/controlled_mix_01_pairAD_ov25_sir0/controlled_mix_01_pairAD_ov25_sir0_speaker_0.wav"
    track_a_spk1 = root / "outputs/first_eval_demo/conv_tasnet/controlled_mix_01_pairAD_ov25_sir0/controlled_mix_01_pairAD_ov25_sir0_speaker_1.wav"
    track_a_spec = "outputs/first_eval_demo/spectrogram_comparison.png"

    # Track B files
    track_b_mix = root / "data/real_world_eval_suite/air_spontaneous_panel_01.wav"
    track_b_spk0 = root / "outputs/real_world_demo/conv_tasnet/air_spontaneous_panel_01/air_spontaneous_panel_01_speaker_0.wav"
    track_b_spk1 = root / "outputs/real_world_demo/conv_tasnet/air_spontaneous_panel_01/air_spontaneous_panel_01_speaker_1.wav"
    track_b_spec = "outputs/real_world_demo/air_spontaneous_panel_01_spectrogram_comparison.png"

    print("Encoding Track A audio stems...")
    a_mix_b64 = encode_audio_base64(track_a_mix)
    a_spk0_b64 = encode_audio_base64(track_a_spk0)
    a_spk1_b64 = encode_audio_base64(track_a_spk1)

    print("Encoding Track B audio stems...")
    b_mix_b64 = encode_audio_base64(track_b_mix)
    b_spk0_b64 = encode_audio_base64(track_b_spk0)
    b_spk1_b64 = encode_audio_base64(track_b_spk1)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Indic Multi-Speaker Separation Demo | First Evaluation</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #090b10;
      --surface: #11151f;
      --surface-border: #1e2638;
      --surface-hover: #171d2b;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --accent-blue: #38bdf8;
      --accent-indigo: #6366f1;
      --accent-emerald: #10b981;
      --accent-rose: #f43f5e;
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 14px;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      background-color: var(--bg);
      color: var(--text-main);
      font-family: var(--font-sans);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
      padding: 24px 16px;
      min-height: 100vh;
      display: flex;
      justify-content: center;
    }}

    .container {{
      width: 100%;
      max-width: 960px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }}

    /* Minimalist Header */
    header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 1px solid var(--surface-border);
      padding-bottom: 16px;
    }}

    .title-group h1 {{
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .title-group h1 span {{
      color: var(--accent-blue);
    }}

    .subtitle {{
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 4px;
    }}

    .badges {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}

    .badge {{
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 500;
      padding: 4px 8px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--surface-border);
      background: var(--surface);
      color: var(--text-muted);
    }}

    .badge.highlight {{
      border-color: rgba(56, 189, 248, 0.4);
      color: var(--accent-blue);
      background: rgba(56, 189, 248, 0.08);
    }}

    .badge.speed {{
      border-color: rgba(16, 185, 129, 0.4);
      color: var(--accent-emerald);
      background: rgba(16, 185, 129, 0.08);
    }}

    /* Track Navigation Tabs */
    .tab-bar {{
      display: flex;
      gap: 6px;
      background: var(--surface);
      padding: 4px;
      border-radius: var(--radius-md);
      border: 1px solid var(--surface-border);
    }}

    .tab-btn {{
      flex: 1;
      padding: 10px 14px;
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-family: var(--font-sans);
      font-size: 13px;
      font-weight: 600;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }}

    .tab-btn:hover {{
      color: #fff;
      background: rgba(255, 255, 255, 0.04);
    }}

    .tab-btn.active {{
      background: #1e2638;
      color: #fff;
      box-shadow: 0 1px 3px rgba(0,0,0,0.4);
    }}

    .tab-btn .tag {{
      font-family: var(--font-mono);
      font-size: 10px;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(255,255,255,0.08);
    }}

    /* Player Card */
    .player-card {{
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-lg);
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}

    .track-meta {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }}

    .track-title {{
      font-size: 16px;
      font-weight: 600;
      color: #fff;
    }}

    .track-desc {{
      font-size: 12px;
      color: var(--text-dim);
      font-family: var(--font-mono);
    }}

    /* Big Minimalist Play Buttons */
    .button-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }}

    .channel-btn {{
      background: #141a27;
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-md);
      padding: 16px 14px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      gap: 6px;
      transition: all 0.15s ease;
      text-align: left;
      position: relative;
      overflow: hidden;
    }}

    .channel-btn:hover {{
      border-color: #334155;
      background: var(--surface-hover);
      transform: translateY(-1px);
    }}

    .channel-btn.active {{
      border-color: var(--accent-blue);
      background: rgba(56, 189, 248, 0.08);
      box-shadow: 0 0 16px rgba(56, 189, 248, 0.15);
    }}

    .channel-btn.active.spk1 {{
      border-color: var(--accent-indigo);
      background: rgba(99, 102, 241, 0.08);
      box-shadow: 0 0 16px rgba(99, 102, 241, 0.15);
    }}

    .btn-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      width: 100%;
    }}

    .btn-role {{
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
    }}

    .channel-btn.active .btn-role {{
      color: var(--accent-blue);
    }}

    .channel-btn.active.spk1 .btn-role {{
      color: var(--accent-indigo);
    }}

    .play-icon {{
      font-size: 14px;
      color: var(--text-muted);
      width: 20px;
      height: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 50%;
      background: rgba(255,255,255,0.06);
    }}

    .channel-btn.active .play-icon {{
      background: var(--accent-blue);
      color: #000;
    }}

    .btn-name {{
      font-size: 14px;
      font-weight: 600;
      color: #fff;
    }}

    .btn-metric {{
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-muted);
    }}

    /* Waveform & Scrubber */
    .visualizer-wrapper {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      background: #0b0e14;
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-md);
      padding: 12px 14px;
    }}

    canvas#oscilloscope {{
      width: 100%;
      height: 52px;
      border-radius: 4px;
      display: block;
    }}

    .scrubber-row {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .time-display {{
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-dim);
      min-width: 75px;
    }}

    .progress-bar-container {{
      flex: 1;
      height: 6px;
      background: #1a2233;
      border-radius: 3px;
      cursor: pointer;
      position: relative;
    }}

    .progress-fill {{
      height: 100%;
      width: 0%;
      background: var(--accent-blue);
      border-radius: 3px;
      transition: width 0.05s linear;
    }}

    /* Transcript Box */
    .transcript-card {{
      background: #0c1017;
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-md);
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}

    .transcript-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    .transcript-text {{
      font-size: 14px;
      color: #e2e8f0;
      line-height: 1.6;
    }}

    .transcript-text .speaker-tag {{
      font-weight: 600;
      margin-right: 6px;
    }}

    .speaker-tag.spk0 {{ color: var(--accent-blue); }}
    .speaker-tag.spk1 {{ color: var(--accent-indigo); }}
    .speaker-tag.mix {{ color: var(--accent-rose); }}

    /* Spectrogram Panel */
    .spectrogram-card {{
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-lg);
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}

    .spectrogram-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .spectrogram-header h3 {{
      font-size: 14px;
      font-weight: 600;
      color: #fff;
    }}

    .spectrogram-img {{
      width: 100%;
      border-radius: var(--radius-sm);
      border: 1px solid var(--surface-border);
      display: block;
      background: #000;
    }}

    /* Minimal Presentation Cheatsheet Drawer */
    details.cheatsheet {{
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: var(--radius-md);
      padding: 12px 16px;
      font-size: 12px;
    }}

    details.cheatsheet summary {{
      cursor: pointer;
      font-weight: 600;
      color: var(--text-muted);
      user-select: none;
    }}

    details.cheatsheet summary:hover {{
      color: #fff;
    }}

    .cheatsheet-content {{
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--surface-border);
      color: var(--text-muted);
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
    }}

    .cue h4 {{
      color: #fff;
      font-size: 12px;
      margin-bottom: 4px;
    }}

    @media (max-width: 640px) {{
      .button-grid {{ grid-template-columns: 1fr; }}
      .cheatsheet-content {{ grid-template-columns: 1fr; }}
      header {{ flex-direction: column; gap: 12px; }}
    }}
  </style>
</head>
<body>

<div class="container">
  <!-- Minimal Header -->
  <header>
    <div class="title-group">
      <h1>Indic Speech Separation <span>Showcase</span></h1>
      <p class="subtitle">Stage 1 Flash Talk Demo & Experimental Verification (Conv-TasNet vs Demucs)</p>
    </div>
    <div class="badges">
      <span class="badge highlight">+15.45 dB SI-SDR</span>
      <span class="badge speed">13.3x Real-Time (0.075 RTF)</span>
      <span class="badge">Apple Silicon MPS</span>
    </div>
  </header>

  <!-- Track Navigation Bar -->
  <nav class="tab-bar">
    <button id="tabTrackA" class="tab-btn active" onclick="switchTrack('A')">
      <span>Track A: Controlled Bilingual Overlap</span>
      <span class="tag">Hero Demo (+21.8 dB)</span>
    </button>
    <button id="tabTrackB" class="tab-btn" onclick="switchTrack('B')">
      <span>Track B: AIR 3-Speaker Panel Debate</span>
      <span class="tag">Real-World Stress Test</span>
    </button>
  </nav>

  <!-- Main Player Card -->
  <main class="player-card">
    <div class="track-meta">
      <div id="trackTitle" class="track-title">controlled_mix_01: Female Hinglish + Male Hindi</div>
      <div id="trackDesc" class="track-desc">Overlap: 25% (1.18s) | SIR: 0 dB (Equal Loudness) | Duration: 9.54s</div>
    </div>

    <!-- 3 Big One-Click Play Buttons -->
    <div class="button-grid">
      <!-- Button 1: Mixture -->
      <button id="btnMix" class="channel-btn" onclick="playChannel('mix')">
        <div class="btn-top">
          <span class="btn-role">Channel: Input</span>
          <span class="play-icon">▶</span>
        </div>
        <div id="lblMix" class="btn-name">Overlapping Mixture</div>
        <div id="metMix" class="btn-metric">Cross-talk collision (0 dB SIR)</div>
      </button>

      <!-- Button 2: Speaker 0 -->
      <button id="btnSpk0" class="channel-btn" onclick="playChannel('spk0')">
        <div class="btn-top">
          <span class="btn-role">Conv-TasNet Ch 0</span>
          <span class="play-icon">▶</span>
        </div>
        <div id="lblSpk0" class="btn-name">Speaker 0: Female Hinglish</div>
        <div id="metSpk0" class="btn-metric">+22.7 dB SI-SDR | Clean</div>
      </button>

      <!-- Button 3: Speaker 1 -->
      <button id="btnSpk1" class="channel-btn spk1" onclick="playChannel('spk1')">
        <div class="btn-top">
          <span class="btn-role">Conv-TasNet Ch 1</span>
          <span class="play-icon">▶</span>
        </div>
        <div id="lblSpk1" class="btn-name">Speaker 1: Male Hindi</div>
        <div id="metSpk1" class="btn-metric">+20.9 dB SI-SDR | Clean</div>
      </button>
    </div>

    <!-- Waveform & Live Visualizer -->
    <div class="visualizer-wrapper">
      <canvas id="oscilloscope"></canvas>
      <div class="scrubber-row">
        <span id="timeElapsed" class="time-display">0:00 / 0:09</span>
        <div class="progress-bar-container" id="progressContainer" onclick="seekAudio(event)">
          <div class="progress-fill" id="progressFill"></div>
        </div>
        <button id="btnStop" onclick="stopAudio()" style="background:transparent; border:none; color:var(--text-muted); cursor:pointer; font-size:12px;">⏹ Stop</button>
      </div>
    </div>

    <!-- Transcript Comparison Box -->
    <div class="transcript-card">
      <div class="transcript-header">
        <span id="transcriptLabel">Active Audio Transcript</span>
        <span id="transcriptStatus" style="font-family:var(--font-mono); font-size:10px; color:var(--accent-emerald);">CLICK A BUTTON TO AUDITION</span>
      </div>
      <div id="transcriptBody" class="transcript-text">
        Click <strong>▶ Play Mixture</strong> to hear the concurrent cross-talk, then click <strong>▶ Play Speaker 0</strong> or <strong>▶ Play Speaker 1</strong> to hear how Conv-TasNet isolated the individual voices cleanly without cross-talk leakage.
      </div>
    </div>
  </main>

  <!-- High-Res Spectrogram Diagnostic -->
  <section class="spectrogram-card">
    <div class="spectrogram-header">
      <h3 id="specTitle">3-Panel Spectrogram Verification (Harmonic Decoupling Proof)</h3>
      <span class="badge">STFT 512 NFFT</span>
    </div>
    <img id="spectrogramImg" class="spectrogram-img" src="{track_a_spec}" alt="Spectrogram Comparison">
  </section>

  <!-- Minimal Presentation Cheat Sheet Drawer -->
  <details class="cheatsheet">
    <summary>Flash Talk 3-Minute Presentation Script & Cue Notes</summary>
    <div class="cheatsheet-content">
      <div class="cue">
        <h4>1. The Acoustic Challenge (0:00 - 0:45)</h4>
        <p>"Indian multi-speaker conversations suffer from dynamic Hinglish code-switching, unconstrained overlap, and lack of ground truth. Without separation, downstream Whisper ASR Word Error Rate jumps to over 65%."</p>
      </div>
      <div class="cue">
        <h4>2. The 10-Second Live Audition (0:45 - 1:45)</h4>
        <p>"Listen to the mixture [Click ▶ Play Mixture]. Both speakers collide. Now listen to Conv-TasNet Channel 0 [Click ▶ Play Speaker 0] and Channel 1 [Click ▶ Play Speaker 1]. Clean, independent streams with +15.45 dB SI-SDR gain."</p>
      </div>
      <div class="cue">
        <h4>3. Stage 2 Transition (1:45 - 3:00)</h4>
        <p>"On Track B [Switch to Track B], Conv-TasNet shows permutation swapping on 3-speaker unsegmented audio. This validates why Stage 2 (Diarization + Continuous Speech Separation) is required in Phase B."</p>
      </div>
    </div>
  </details>
</div>

<!-- Embedded Audio Stems (Base64 Data URIs) -->
<script>
  const AUDIO_DATA = {{
    A: {{
      mix: "{a_mix_b64}",
      spk0: "{a_spk0_b64}",
      spk1: "{a_spk1_b64}",
      spec: "{track_a_spec}",
      title: "controlled_mix_01: Female Hinglish + Male Hindi",
      desc: "Overlap: 25% (1.18s) | SIR: 0 dB (Equal Loudness) | Duration: 9.54s",
      labels: {{
        mix: "Overlapping Mixture",
        spk0: "Speaker 0: Female Hinglish",
        spk1: "Speaker 1: Male Hindi"
      }},
      metrics: {{
        mix: "Cross-talk collision (0 dB SIR)",
        spk0: "+22.7 dB SI-SDR | Clean",
        spk1: "+20.9 dB SI-SDR | Clean"
      }},
      transcripts: {{
        mix: "<span class='speaker-tag mix'>[COLLISION]</span> Both speakers talking simultaneously during 3.5s - 4.7s. Standard ASR fails to transcribe the overlapping portion.",
        spk0: "<span class='speaker-tag spk0'>Speaker 0 (Female Hinglish):</span> 'उन्हें दस दिन तक rehab करना होगा और उसके बाद उनका fitness test लिया जाएगा।'",
        spk1: "<span class='speaker-tag spk1'>Speaker 1 (Male Hindi):</span> 'यह एक वॉइस ऑडियो डेटासेट है जिसे आप AI प्रशिक्षण उद्देश्यों के लिए उपयोग कर सकते हैं।'"
      }}
    }},
    B: {{
      mix: "{b_mix_b64}",
      spk0: "{b_spk0_b64}",
      spk1: "{b_spk1_b64}",
      spec: "{track_b_spec}",
      title: "air_spontaneous_panel_01: AIR 3-Speaker Panel Debate",
      desc: "3 Talkers (Host + 2 Guests) | Conversational Barge-Ins | Duration: 13.02s",
      labels: {{
        mix: "3-Speaker Debate Mixture",
        spk0: "Channel 0: Dominant Stream",
        spk1: "Channel 1: Interjection Stream"
      }},
      metrics: {{
        mix: "Spontaneous conversational collisions",
        spk0: "Isolates host & main dialogue",
        spk1: "Isolates overlapping barge-in"
      }},
      transcripts: {{
        mix: "<span class='speaker-tag mix'>[MULTI-PARTY]</span> Host introduces rehab, Guest 1 interjects with financial scheme at 3.8s, Guest 2 interjects with smartphone software at 6.8s.",
        spk0: "<span class='speaker-tag spk0'>Channel 0 (Host + Guest 2):</span> 'उन्हें दस दिन तक rehab करना होगा... smartphone में विशेष तौर से hardware और software...'",
        spk1: "<span class='speaker-tag spk1'>Channel 1 (Guest 1 Barge-In):</span> 'इस योजना के अंतर्गत सभी महिलाओं को विशेष वित्तीय सहायता दी जाएगी।'"
      }}
    }}
  }};

  let currentTrack = 'A';
  let activeChannel = null;
  let audioElem = new Audio();
  let audioCtx = null;
  let analyser = null;
  let sourceNode = null;
  let animFrameId = null;

  function initAudioContext() {{
    if (!audioCtx) {{
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioContext();
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      sourceNode = audioCtx.createMediaElementSource(audioElem);
      sourceNode.connect(analyser);
      analyser.connect(audioCtx.destination);
    }}
    if (audioCtx.state === 'suspended') {{
      audioCtx.resume();
    }}
  }}

  function switchTrack(trackKey) {{
    stopAudio();
    currentTrack = trackKey;
    document.getElementById('tabTrackA').classList.toggle('active', trackKey === 'A');
    document.getElementById('tabTrackB').classList.toggle('active', trackKey === 'B');

    const data = AUDIO_DATA[trackKey];
    document.getElementById('trackTitle').textContent = data.title;
    document.getElementById('trackDesc').textContent = data.desc;
    document.getElementById('lblMix').textContent = data.labels.mix;
    document.getElementById('lblSpk0').textContent = data.labels.spk0;
    document.getElementById('lblSpk1').textContent = data.labels.spk1;
    document.getElementById('metMix').textContent = data.metrics.mix;
    document.getElementById('metSpk0').textContent = data.metrics.spk0;
    document.getElementById('metSpk1').textContent = data.metrics.spk1;
    document.getElementById('spectrogramImg').src = data.spec;
    document.getElementById('transcriptBody').innerHTML = "Switched to " + (trackKey === 'A' ? "Track A (Controlled)" : "Track B (Real-World)") + ". Click any button above to play.";
    document.getElementById('transcriptStatus').textContent = "READY";
  }}

  function playChannel(channelKey) {{
    initAudioContext();

    // Reset button states
    ['btnMix', 'btnSpk0', 'btnSpk1'].forEach(id => {{
      document.getElementById(id).classList.remove('active');
      document.getElementById(id).querySelector('.play-icon').textContent = '▶';
    }});

    const btnId = channelKey === 'mix' ? 'btnMix' : (channelKey === 'spk0' ? 'btnSpk0' : 'btnSpk1');
    const activeBtn = document.getElementById(btnId);

    if (activeChannel === channelKey && !audioElem.paused) {{
      audioElem.pause();
      activeBtn.querySelector('.play-icon').textContent = '▶';
      activeChannel = null;
      document.getElementById('transcriptStatus').textContent = "PAUSED";
      return;
    }}

    activeChannel = channelKey;
    activeBtn.classList.add('active');
    activeBtn.querySelector('.play-icon').textContent = '⏸';

    const trackData = AUDIO_DATA[currentTrack];
    audioElem.src = trackData[channelKey];
    audioElem.play().then(() => {{
      document.getElementById('transcriptStatus').textContent = "NOW PLAYING: " + trackData.labels[channelKey].toUpperCase();
      document.getElementById('transcriptBody').innerHTML = trackData.transcripts[channelKey];
      drawOscilloscope();
    }}).catch(err => console.error(err));
  }}

  function stopAudio() {{
    audioElem.pause();
    audioElem.currentTime = 0;
    activeChannel = null;
    ['btnMix', 'btnSpk0', 'btnSpk1'].forEach(id => {{
      document.getElementById(id).classList.remove('active');
      document.getElementById(id).querySelector('.play-icon').textContent = '▶';
    }});
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('transcriptStatus').textContent = "STOPPED";
  }}

  audioElem.addEventListener('timeupdate', () => {{
    if (!audioElem.duration) return;
    const progress = (audioElem.currentTime / audioElem.duration) * 100;
    document.getElementById('progressFill').style.width = progress + '%';
    const curMin = Math.floor(audioElem.currentTime / 60);
    const curSec = Math.floor(audioElem.currentTime % 60).toString().padStart(2, '0');
    const durMin = Math.floor(audioElem.duration / 60);
    const durSec = Math.floor(audioElem.duration % 60).toString().padStart(2, '0');
    document.getElementById('timeElapsed').textContent = `${{curMin}}:${{curSec}} / ${{durMin}}:${{durSec}}`;
  }});

  audioElem.addEventListener('ended', () => {{
    stopAudio();
  }});

  function seekAudio(e) {{
    if (!audioElem.duration) return;
    const rect = document.getElementById('progressContainer').getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    audioElem.currentTime = pos * audioElem.duration;
  }}

  // Canvas Oscilloscope / Frequency Visualizer
  const canvas = document.getElementById('oscilloscope');
  const ctx = canvas.getContext('2d');

  function resizeCanvas() {{
    canvas.width = canvas.clientWidth * window.devicePixelRatio;
    canvas.height = canvas.clientHeight * window.devicePixelRatio;
  }}
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  function drawOscilloscope() {{
    if (!analyser || audioElem.paused) {{
      drawIdleWave();
      return;
    }}
    animFrameId = requestAnimationFrame(drawOscilloscope);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(dataArray);

    ctx.fillStyle = '#0b0e14';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2 * window.devicePixelRatio;
    ctx.strokeStyle = activeChannel === 'spk1' ? '#6366f1' : (activeChannel === 'spk0' ? '#38bdf8' : '#f43f5e');
    ctx.beginPath();

    const sliceWidth = canvas.width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {{
      const v = dataArray[i] / 128.0;
      const y = (v * canvas.height) / 2;

      if (i === 0) {{
        ctx.moveTo(x, y);
      }} else {{
        ctx.lineTo(x, y);
      }}
      x += sliceWidth;
    }}

    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  }}

  function drawIdleWave() {{
    ctx.fillStyle = '#0b0e14';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 1 * window.devicePixelRatio;
    ctx.strokeStyle = '#1e2638';
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  }}

  drawIdleWave();
</script>

</body>
</html>
"""

    demo_path = root / "demo.html"
    with open(demo_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[✓] Successfully compiled minimalist dark-mode showcase player: {demo_path}")
    print(f"    File size: {demo_path.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
