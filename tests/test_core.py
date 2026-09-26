"""
Unit tests for Core Scaffolding, BaseModule, and PipelineState.
"""

import json
from pathlib import Path
import pytest
import torch

from src.core.base_module import BaseModule, resolve_device
from src.core.state import (
    AudioMetadata,
    DiarizationSegment,
    DiarizationTimeline,
    KeyFinding,
    PipelineState,
    PostProcessedReport,
    RawTranscript,
    SeparatedTrack,
    SeparationResult,
    SpeakerDialogue,
    Utterance,
)


class DummyMockModule(BaseModule):
    """Concrete mock module to test BaseModule lifecycle and telemetry."""

    def initialize(self) -> None:
        self.initialized = True

    def process(self, state: PipelineState) -> PipelineState:
        state.errors.append("dummy_processed")
        return state


def test_resolve_device():
    device = resolve_device("auto")
    assert isinstance(device, torch.device)
    if torch.backends.mps.is_available():
        assert device.type == "mps"
    elif torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_base_module_lifecycle():
    module = DummyMockModule()
    state = PipelineState(input_audio_path="test_audio.wav")
    
    # Process through __call__ to trigger telemetry
    result_state = module(state)
    assert "dummy_processed" in result_state.errors
    assert "DummyMockModule" in result_state.execution_telemetry
    assert "elapsed_seconds" in result_state.execution_telemetry["DummyMockModule"]
    assert "device" in result_state.execution_telemetry["DummyMockModule"]


def test_pipeline_state_serialization(tmp_path: Path):
    state = PipelineState(
        input_audio_path="samples/test.wav",
        audio_metadata=AudioMetadata(
            source_file="test.wav",
            duration_seconds=12.5,
            rms_energy_db=-21.5,
        ),
        separation_result=SeparationResult(
            input_audio_path="samples/test.wav",
            num_sources_detected=2,
            isolated_tracks=[
                SeparatedTrack(source_index=0, track_path="stems/s0.wav", energy_ratio=0.6),
                SeparatedTrack(source_index=1, track_path="stems/s1.wav", energy_ratio=0.4),
            ],
        ),
        diarization_timeline=DiarizationTimeline(
            audio_file="samples/test.wav",
            total_speakers_detected=2,
            segments=[
                DiarizationSegment(segment_id=0, speaker_id="SPEAKER_00", start_time=0.0, end_time=5.0),
                DiarizationSegment(segment_id=1, speaker_id="SPEAKER_01", start_time=5.5, end_time=12.0),
            ],
        ),
        raw_transcript=RawTranscript(
            utterances=[
                Utterance(
                    utterance_id=0,
                    speaker_id="SPEAKER_00",
                    start_time=0.0,
                    end_time=5.0,
                    raw_text="नमस्ते",
                )
            ]
        ),
        final_report=PostProcessedReport(
            executive_summary="Summary of discussion.",
            primary_topics=["Agriculture"],
            keyword_tags=["#Kisan"],
            key_findings=[
                KeyFinding(topic="MSP", detail="Price is above MSP", speaker_attribution="SPEAKER_00")
            ],
            speaker_separated_dialogue=[
                SpeakerDialogue(speaker="SPEAKER_00", timestamp="00:00 - 00:05", cleaned_text="नमस्ते।")
            ],
        ),
    )

    json_file = tmp_path / "pipeline_state.json"
    state.save_json(str(json_file))
    assert json_file.exists()

    loaded_state = PipelineState.load_json(str(json_file))
    assert loaded_state.input_audio_path == "samples/test.wav"
    assert loaded_state.audio_metadata.duration_seconds == 12.5
    assert len(loaded_state.separation_result.isolated_tracks) == 2
    assert len(loaded_state.diarization_timeline.segments) == 2
    assert loaded_state.final_report.key_findings[0].topic == "MSP"
