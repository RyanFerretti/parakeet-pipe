# Speaker diarization module for Parakeet
# This module integrates pyannote.audio for speaker identification

from typing import Dict, List, Optional, Tuple, Union
import os
import logging
import tempfile
import numpy as np
import torch
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class SpeakerSegment(BaseModel):
    """A segment of speech from a specific speaker"""
    start: float
    end: float
    speaker: str

class DiarizationResult(BaseModel):
    """Result of speaker diarization"""
    segments: List[SpeakerSegment]
    num_speakers: int

class Diarizer:
    """Speaker diarization using pyannote.audio"""

    def __init__(self, access_token: Optional[str] = None):
        self.pipeline = None
        self.access_token = access_token

        # Prefer Apple Silicon (MPS) when available, then CUDA, otherwise CPU.
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self._initialize()

    def _initialize(self):
        """Initialize the diarization pipeline"""
        try:
            from pyannote.audio import Pipeline

            if not self.access_token:
                logger.warning("No access token provided. Using HUGGINGFACE_ACCESS_TOKEN environment variable.")
                self.access_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

            if not self.access_token:
                logger.error("No access token available. Diarization will not work.")
                return

            # Initialize the pipeline
            try:
                # Newer pyannote builds expect the `token` keyword (Community-1 docs)
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    token=self.access_token
                )
            except TypeError:
                # Fall back to older signature to avoid breaking existing envs
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    use_auth_token=self.access_token
                )

            # Move to GPU if available
            self.pipeline.to(torch.device(self.device))
            logger.info(f"Diarization pipeline initialized on {self.device}")

        except ImportError:
            logger.error("Failed to import pyannote.audio. Please install it with 'pip install pyannote.audio'")
        except Exception as e:
            logger.error(f"Failed to initialize diarization pipeline: {str(e)}")

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None) -> DiarizationResult:
        """
        Perform speaker diarization on an audio file

        Args:
            audio_path: Path to the audio file
            num_speakers: Optional number of speakers (if known)

        Returns:
            DiarizationResult with speaker segments
        """
        if self.pipeline is None:
            logger.error("Diarization pipeline not initialized")
            return DiarizationResult(segments=[], num_speakers=0)

        try:
            # Run the diarization pipeline
            diarization_output = self.pipeline(
                audio_path,
                num_speakers=num_speakers
            )

            # Community-1 exposes both regular and exclusive diarization tracks.
            annotation = getattr(diarization_output, "exclusive_speaker_diarization", None)
            if annotation is None:
                annotation = getattr(diarization_output, "speaker_diarization", diarization_output)

            # Convert to our format
            segments = []
            speakers = set()

            # Process the diarization result
            for track in annotation.itertracks(yield_label=True):
                turn = track[0]
                speaker = track[2] if len(track) > 2 else track[1]
                # Convert speaker label to consistent format
                # This handles different formats from pyannote.audio versions
                if isinstance(speaker, str) and not speaker.startswith("SPEAKER_"):
                    speaker_id = f"SPEAKER_{speaker}"
                else:
                    speaker_id = speaker

                segments.append(SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=f"speaker_{speaker_id}"
                ))
                speakers.add(speaker_id)

            # Sort segments by start time
            segments.sort(key=lambda x: x.start)

            return DiarizationResult(
                segments=segments,
                num_speakers=len(speakers)
            )

        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}")
            return DiarizationResult(segments=[], num_speakers=0)

    def merge_with_transcription(self,
                                diarization: DiarizationResult,
                                transcription_segments: list) -> list:
        """
        Merge diarization results with transcription segments

        Args:
            diarization: Speaker diarization result
            transcription_segments: List of transcription segments with start/end times

        Returns:
            Merged list of segments with speaker information
        """
        return merge_diarization_with_transcription(diarization, transcription_segments)


def merge_diarization_with_transcription(
    diarization: DiarizationResult,
    transcription_segments: List,
) -> List:
    """
    Assign speakers from diarization output to transcription segments.
    """
    if not diarization.segments or not transcription_segments:
        return transcription_segments

    for segment in transcription_segments:
        start = segment.start
        end = segment.end
        overlapping = []

        for spk_segment in diarization.segments:
            overlap_start = max(start, spk_segment.start)
            overlap_end = min(end, spk_segment.end)

            if overlap_end > overlap_start:
                duration = overlap_end - overlap_start
                overlapping.append((spk_segment.speaker, duration))

        if overlapping:
            overlapping.sort(key=lambda x: x[1], reverse=True)
            setattr(segment, "speaker", overlapping[0][0])
        else:
            setattr(segment, "speaker", "unknown")

    return transcription_segments


def apply_speaker_labels_to_text(segments: List) -> None:
    """
    Prefix segment text with speaker labels for readability.
    """
    previous_speaker = None
    seen_speakers = set()

    for segment in segments:
        speaker_label = getattr(segment, "speaker", None)
        if not speaker_label:
            continue

        if speaker_label.startswith("speaker_"):
            try:
                parts = speaker_label.split("_")
                speaker_num = int(parts[-1]) + 1
                if speaker_label != previous_speaker:
                    if speaker_label not in seen_speakers:
                        prefix = f"Speaker {speaker_num}: "
                        seen_speakers.add(speaker_label)
                    else:
                        prefix = f"{speaker_num}: "
                    segment.text = f"{prefix}{segment.text}"
                previous_speaker = speaker_label
            except (ValueError, IndexError):
                if "Speaker" != previous_speaker:
                    segment.text = f"Speaker: {segment.text}"
                    previous_speaker = "Speaker"
