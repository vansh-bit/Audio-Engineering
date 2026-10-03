"""
Generates an ultra-minimalist, clean, white, professional HTML5 Audio Showcase Player (demo.html).
No transcripts, no cue notes, no noisy widgets. Pure, high-precision audio auditioning for:
1. Controlled Environment (Overlapping Mix vs Separated Stems)
2. Real-World Environment (Overlapping Mix vs Separated Stems)
Embeds audio as base64 data URIs for 100% offline, zero-CORS instant playback.
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

    # Track A (Controlled)
    track_a_mix = root / "data/controlled_eval_suite/synthetic_mixtures/controlled_mix_01_pairAD_ov25_sir0.wav"
    track_a_spk0 = root / "outputs/first_eval_demo/conv_tasnet/controlled_mix_01_pairAD_ov25_sir0/controlled_mix_01_pairAD_ov25_sir0_speaker_0.wav"
    track_a_spk1 = root / "outputs/first_eval_demo/conv_tasnet/controlled_mix_01_pairAD_ov25_sir0/controlled_mix_01_pairAD_ov25_sir0_speaker_1.wav"
    track_a_spec = "outputs/first_eval_demo/spectrogram_comparison.png"

    # Track B (Real-World)
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
  <title>Speech Separation Evaluation | Conv-TasNet Demo</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #ffffff;
      --card-bg: #f8fafc;
      --border: #e2e8f0;
      --border-focus: #cbd5e1;
      --text-primary: #0f172a;
      --text-secondary: #475569;
      --text-muted: #64748b;
      --accent: #2563eb;
      --active-dark: #0f172a;
      --active-text: #ffffff;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      background-color: var(--bg);
      color: var(--text-primary);
      font-family: var(--font-sans);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
      padding: 40px 20px;
      min-height: 100vh;
      display: flex;
      justify-content: center;
    }}

    .container {{
      width: 100%;
      max-width: 820px;
      display: flex;
      flex-direction: column;
      gap: 28px;
    }}

    /* Minimalist Header */
    header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      border-bottom: 1px solid var(--border);
      padding-bottom: 20px;
    }}

    .header-title h1 {{
      font-size: 20px;
      font-weight: 700;
      color: var(--text-primary);
      letter-spacing: -0.02em;
    }}

    .header-title p {{
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 4px;
    }}

    .header-metric {{
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 500;
      color: var(--text-secondary);
      background: var(--card-bg);
      padding: 6px 12px;
      border: 1px solid var(--border);
      border-radius: 6px;
    }}

    /* Clean Segmented Control */
    .segmented-control {{
      display: flex;
      background: var(--card-bg);
      padding: 4px;
      border-radius: 8px;
      border: 1px solid var(--border);
      gap: 4px;
    }}

    .seg-btn {{
      flex: 1;
      padding: 10px 16px;
      background: transparent;
      border: none;
      color: var(--text-secondary);
      font-family: var(--font-sans);
      font-size: 13px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
      text-align: center;
    }}

    .seg-btn:hover {{
      color: var(--text-primary);
    }}

    .seg-btn.active {{
      background: #ffffff;
      color: var(--text-primary);
      box-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    }}

    /* Environment Card */
    .player-card {{
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }}

    .meta-bar {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }}

    .env-name {{
      font-size: 15px;
      font-weight: 600;
      color: var(--text-primary);
    }}

    .env-details {{
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-muted);
    }}

    /* Clean 3-Button Grid */
    .audio-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }}

    .audio-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      gap: 8px;
      text-align: left;
      transition: all 0.12s ease;
      position: relative;
    }}

    .audio-card:hover {{
      border-color: var(--border-focus);
      background: #f1f5f9;
    }}

    .audio-card.active {{
      background: var(--active-dark);
      border-color: var(--active-dark);
      color: var(--active-text);
    }}

    .card-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}

    .card-label {{
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }}

    .audio-card.active .card-label {{
      color: #94a3b8;
    }}

    .play-badge {{
      font-size: 12px;
      width: 22px;
      height: 22px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 50%;
      background: #ffffff;
      color: var(--text-primary);
      border: 1px solid var(--border);
    }}

    .audio-card.active .play-badge {{
      background: #ffffff;
      color: #000000;
      border: none;
    }}

    .card-name {{
      font-size: 13px;
      font-weight: 600;
      color: var(--text-primary);
    }}

    .audio-card.active .card-name {{
      color: #ffffff;
    }}

    .card-sub {{
      font-size: 11px;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }}

    .audio-card.active .card-sub {{
      color: #cbd5e1;
    }}

    /* Minimalist Scrubber & Timeline */
    .scrubber-panel {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding-top: 8px;
      border-top: 1px solid var(--border);
    }}

    .timeline-row {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}

    .time-readout {{
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-muted);
      min-width: 80px;
    }}

    .progress-track {{
      flex: 1;
      height: 4px;
      background: #e2e8f0;
      border-radius: 2px;
      cursor: pointer;
      position: relative;
    }}

    .progress-fill {{
      height: 100%;
      width: 0%;
      background: var(--active-dark);
      border-radius: 2px;
      transition: width 0.05s linear;
    }}

    .now-playing-status {{
      font-size: 11px;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }}

    .stop-btn {{
      background: none;
      border: none;
      font-family: var(--font-sans);
      font-size: 11px;
      font-weight: 500;
      color: var(--text-secondary);
      cursor: pointer;
      padding: 2px 6px;
    }}

    .stop-btn:hover {{
      color: #000;
    }}

    /* Minimal Spectrogram Section */
    .spectrogram-section {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}

    .spec-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }}

    .spec-header h3 {{
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
    }}

    .spec-view {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      display: block;
      background: #ffffff;
    }}

    @media (max-width: 600px) {{
      .audio-grid {{ grid-template-columns: 1fr; }}
      header {{ flex-direction: column; gap: 8px; }}
    }}
  </style>
</head>
<body>

<div class="container">
  <!-- Minimal Header -->
  <header>
    <div class="header-title">
      <h1>Audio Source Separation</h1>
      <p>Conv-TasNet Comparative Evaluation (Track A & Track B)</p>
    </div>
    <div class="header-metric">
      SI-SDR: +15.45 dB | RTF: 0.075 (13x)
    </div>
  </header>

  <!-- Clean Environment Segmented Control -->
  <div class="segmented-control">
    <button id="btnTabA" class="seg-btn active" onclick="switchEnv('A')">
      Controlled Environment (2-Speaker Overlap)
    </button>
    <button id="btnTabB" class="seg-btn" onclick="switchEnv('B')">
      Real-World Environment (3-Speaker Debate)
    </button>
  </div>

  <!-- Player Card -->
  <div class="player-card">
    <div class="meta-bar">
      <span id="envTitle" class="env-name">Female Hinglish + Male Hindi</span>
      <span id="envDetails" class="env-details">25% Overlap | 0 dB SIR | 9.54s</span>
    </div>

    <!-- 3 Clean Playback Cards -->
    <div class="audio-grid">
      <!-- 1. Overlapping Mixture -->
      <div id="cardMix" class="audio-card" onclick="toggleAudio('mix')">
        <div class="card-top">
          <span class="card-label">Input</span>
          <span class="play-badge" id="iconMix">▶</span>
        </div>
        <div class="card-name" id="nameMix">Overlapping Mixture</div>
        <div class="card-sub" id="subMix">Combined cross-talk</div>
      </div>

      <!-- 2. Separated Speaker 0 -->
      <div id="cardSpk0" class="audio-card" onclick="toggleAudio('spk0')">
        <div class="card-top">
          <span class="card-label">Separated</span>
          <span class="play-badge" id="iconSpk0">▶</span>
        </div>
        <div class="card-name" id="nameSpk0">Speaker 1</div>
        <div class="card-sub" id="subSpk0">Channel 0 (Isolated)</div>
      </div>

      <!-- 3. Separated Speaker 1 -->
      <div id="cardSpk1" class="audio-card" onclick="toggleAudio('spk1')">
        <div class="card-top">
          <span class="card-label">Separated</span>
          <span class="play-badge" id="iconSpk1">▶</span>
        </div>
        <div class="card-name" id="nameSpk1">Speaker 2</div>
        <div class="card-sub" id="subSpk1">Channel 1 (Isolated)</div>
      </div>
    </div>

    <!-- Scrubber & Timeline -->
    <div class="scrubber-panel">
      <div class="timeline-row">
        <span id="timeReadout" class="time-readout">0:00 / 0:09</span>
        <div class="progress-track" id="progressTrack" onclick="seekAudio(event)">
          <div class="progress-fill" id="progressFill"></div>
        </div>
        <button class="stop-btn" onclick="stopAudio()">Stop</button>
      </div>
      <div id="nowPlayingText" class="now-playing-status">Ready. Click any channel above to play.</div>
    </div>
  </div>

  <!-- Minimal Spectrogram -->
  <div class="spectrogram-section">
    <div class="spec-header">
      <span class="card-label">Acoustic Diagnostic</span>
      <span class="card-label">Mixture vs Separated Channels</span>
    </div>
    <img id="spectrogramImg" class="spec-view" src="{track_a_spec}" alt="Spectrogram Comparison">
  </div>
</div>

<script>
  const AUDIO_DATA = {{
    A: {{
      mix: "{a_mix_b64}",
      spk0: "{a_spk0_b64}",
      spk1: "{a_spk1_b64}",
      spec: "{track_a_spec}",
      title: "Controlled Mixture: Female Hinglish + Male Hindi",
      details: "25% Overlap | 0 dB SIR | 9.54s",
      nameMix: "Overlapping Mixture",
      nameSpk0: "Speaker 1 (Female)",
      nameSpk1: "Speaker 2 (Male)",
      subMix: "Concurrent cross-talk",
      subSpk0: "Channel 0 (+22.7 dB SI-SDR)",
      subSpk1: "Channel 1 (+20.9 dB SI-SDR)"
    }},
    B: {{
      mix: "{b_mix_b64}",
      spk0: "{b_spk0_b64}",
      spk1: "{b_spk1_b64}",
      spec: "{track_b_spec}",
      title: "Real-World Environment: AIR 3-Speaker Panel Debate",
      details: "3 Speakers | Spontaneous Cross-Talk | 13.02s",
      nameMix: "3-Speaker Mixture",
      nameSpk0: "Channel 0 (Host)",
      nameSpk1: "Channel 1 (Guest)",
      subMix: "Live conversational debate",
      subSpk0: "Dominant speaker isolated",
      subSpk1: "Overlapping barge-in isolated"
    }}
  }};

  let currentEnv = 'A';
  let currentKey = null;
  let audio = new Audio();

  function switchEnv(env) {{
    stopAudio();
    currentEnv = env;
    document.getElementById('btnTabA').classList.toggle('active', env === 'A');
    document.getElementById('btnTabB').classList.toggle('active', env === 'B');

    const d = AUDIO_DATA[env];
    document.getElementById('envTitle').textContent = d.title;
    document.getElementById('envDetails').textContent = d.details;
    document.getElementById('nameMix').textContent = d.nameMix;
    document.getElementById('nameSpk0').textContent = d.nameSpk0;
    document.getElementById('nameSpk1').textContent = d.nameSpk1;
    document.getElementById('subMix').textContent = d.subMix;
    document.getElementById('subSpk0').textContent = d.subSpk0;
    document.getElementById('subSpk1').textContent = d.subSpk1;
    document.getElementById('spectrogramImg').src = d.spec;
    document.getElementById('nowPlayingText').textContent = "Ready. Click any channel above to play.";
  }}

  function toggleAudio(key) {{
    const cards = {{ mix: 'cardMix', spk0: 'cardSpk0', spk1: 'cardSpk1' }};
    const icons = {{ mix: 'iconMix', spk0: 'iconSpk0', spk1: 'iconSpk1' }};

    // If currently playing this same audio, toggle pause
    if (currentKey === key && !audio.paused) {{
      audio.pause();
      document.getElementById(icons[key]).textContent = '▶';
      document.getElementById('nowPlayingText').textContent = "Paused";
      return;
    }}

    // Reset all cards
    Object.keys(cards).forEach(k => {{
      document.getElementById(cards[k]).classList.remove('active');
      document.getElementById(icons[k]).textContent = '▶';
    }});

    // Activate selected
    currentKey = key;
    document.getElementById(cards[key]).classList.add('active');
    document.getElementById(icons[key]).textContent = '⏸';

    const d = AUDIO_DATA[currentEnv];
    audio.src = d[key];
    audio.play().then(() => {{
      const label = key === 'mix' ? d.nameMix : (key === 'spk0' ? d.nameSpk0 : d.nameSpk1);
      document.getElementById('nowPlayingText').textContent = "Playing: " + label;
    }}).catch(e => console.error(e));
  }}

  function stopAudio() {{
    audio.pause();
    audio.currentTime = 0;
    currentKey = null;
    ['mix', 'spk0', 'spk1'].forEach(k => {{
      const card = document.getElementById(k === 'mix' ? 'cardMix' : (k === 'spk0' ? 'cardSpk0' : 'cardSpk1'));
      const icon = document.getElementById(k === 'mix' ? 'iconMix' : (k === 'spk0' ? 'iconSpk0' : 'iconSpk1'));
      if (card) card.classList.remove('active');
      if (icon) icon.textContent = '▶';
    }});
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('nowPlayingText').textContent = "Stopped";
  }}

  audio.addEventListener('timeupdate', () => {{
    if (!audio.duration) return;
    const progress = (audio.currentTime / audio.duration) * 100;
    document.getElementById('progressFill').style.width = progress + '%';

    const curM = Math.floor(audio.currentTime / 60);
    const curS = Math.floor(audio.currentTime % 60).toString().padStart(2, '0');
    const durM = Math.floor(audio.duration / 60);
    const durS = Math.floor(audio.duration % 60).toString().padStart(2, '0');
    document.getElementById('timeReadout').textContent = `${{curM}}:${{curS}} / ${{durM}}:${{durS}}`;
  }});

  audio.addEventListener('ended', () => {{
    stopAudio();
  }});

  function seekAudio(e) {{
    if (!audio.duration) return;
    const rect = document.getElementById('progressTrack').getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    audio.currentTime = pos * audio.duration;
  }}
</script>

</body>
</html>
"""

    demo_path = root / "demo.html"
    with open(demo_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[✓] Successfully generated clean, white, minimalist showcase: {demo_path}")
    print(f"    File size: {demo_path.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
