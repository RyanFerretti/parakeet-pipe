import logging
import os
from pathlib import Path
from typing import Optional, List

from audio import convert_audio_to_wav, split_audio_into_chunks
from config import Config, get_config
from diarization import (
    Diarizer,
    apply_speaker_labels_to_text,
    merge_diarization_with_transcription,
)
from models import TranscriptionResponse, WhisperSegment
from transcription import (
    load_model,
    transcribe_audio_chunk,
)

logger = logging.getLogger(__name__)


class TranscriptionService:
    """
    Shared service that powers both the HTTP API and the CLI workflow.
    Handles model loading, chunking, transcription, and optional diarization.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self._asr_model = None
        self._diarizer = None

    def ensure_model_loaded(self):
        """Load the Parakeet model if it has not been loaded yet."""
        if self._asr_model is None:
            model_id = self.config.model_id
            logger.info(f"Loading model {model_id}")
            self._asr_model = load_model(model_id)
            logger.info(f"Model {model_id} loaded successfully")
        return self._asr_model

    def is_model_loaded(self) -> bool:
        return self._asr_model is not None

    def _get_or_create_diarizer(self) -> Optional[Diarizer]:
        if self._diarizer is not None:
            return self._diarizer

        hf_token = self.config.get_hf_token()
        if not hf_token:
            return None

        self._diarizer = Diarizer(access_token=hf_token)
        return self._diarizer

    def transcribe_file(
        self,
        input_path: str,
        *,
        language: Optional[str] = None,
        diarize: Optional[bool] = None,
        include_diarization_in_text: Optional[bool] = None,
        word_timestamps: bool = False,
        return_segments: bool = False,
    ) -> TranscriptionResponse:
        """
        Transcribe the provided audio file using Parakeet, optionally performing diarization.
        """
        model = self.ensure_model_loaded()

        diarize = (
            self.config.enable_diarization if diarize is None else bool(diarize)
        )
        include_diarization_in_text = (
            self.config.include_diarization_in_text
            if include_diarization_in_text is None
            else bool(include_diarization_in_text)
        )

        temp_dir = Path(self.config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        artifacts_to_cleanup: List[str] = []
        audio_chunks: List[str] = []

        try:
            # Convert to WAV and chunk if necessary
            wav_file = convert_audio_to_wav(str(input_path))
            artifacts_to_cleanup.append(wav_file)

            audio_chunks = split_audio_into_chunks(
                wav_file, chunk_duration=self.config.chunk_duration
            )

            for chunk in audio_chunks:
                if chunk != wav_file:
                    artifacts_to_cleanup.append(chunk)

            # Setup diarization if requested
            diarizer = None
            diarization_result = None
            if diarize:
                diarizer = self._get_or_create_diarizer()
                if diarizer:
                    logger.info("Performing speaker diarization")
                    diarization_result = diarizer.diarize(wav_file)
                else:
                    logger.warning(
                        "Diarization requested but no HuggingFace token is available"
                    )

            # Process chunks
            all_text: List[str] = []
            all_segments: List[WhisperSegment] = []

            for i, chunk_path in enumerate(audio_chunks):
                logger.info(f"Processing chunk {i+1}/{len(audio_chunks)}")
                chunk_text, chunk_segments = transcribe_audio_chunk(
                    model,
                    chunk_path,
                    language=language,
                    word_timestamps=word_timestamps,
                )

                # Apply chunk offset to segment timestamps
                if i > 0:
                    offset = i * self.config.chunk_duration
                    for segment in chunk_segments:
                        segment.start += offset
                        segment.end += offset

                all_text.append(chunk_text)
                all_segments.extend(chunk_segments)

            full_text = " ".join(t for t in all_text if t)

            # Merge diarization info
            if diarizer and diarization_result and diarization_result.segments:
                logger.info(
                    f"Diarization found {diarization_result.num_speakers} speakers"
                )
                all_segments = merge_diarization_with_transcription(
                    diarization_result, all_segments
                )

                if include_diarization_in_text:
                    apply_speaker_labels_to_text(all_segments)
                    full_text = " ".join(segment.text for segment in all_segments)
            elif diarize:
                logger.warning("Diarization not applied or returned no speakers")

            response = TranscriptionResponse(
                text=full_text,
                segments=all_segments if return_segments else None,
                language=language,
                duration=self._estimate_duration(all_segments),
                model=self.config.model_id,
            )

            return response

        finally:
            for artifact in artifacts_to_cleanup:
                if artifact and os.path.exists(artifact):
                    try:
                        os.unlink(artifact)
                    except OSError:
                        logger.debug(f"Failed to remove temp file {artifact}")

    @staticmethod
    def _estimate_duration(segments: List[WhisperSegment]) -> float:
        if not segments:
            return 0.0
        return sum(len(segment.text.split()) for segment in segments) / 150

