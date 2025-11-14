import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

from diarization import (
    DiarizationResult,
    SpeakerSegment,
    apply_speaker_labels_to_text,
    merge_diarization_with_transcription,
)
from models import TranscriptionResponse, WhisperSegment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge transcription segments with diarization output."
    )
    parser.add_argument(
        "--segments",
        required=True,
        help="JSON file containing transcription segments (from the ASR CLI).",
    )
    parser.add_argument(
        "--speakers",
        required=True,
        help="JSON file produced by diarize_cli.py with speaker segments.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Destination path for the merged verbose_json output.",
    )
    parser.add_argument(
        "--include-speakers-in-text",
        action="store_true",
        help="Prefix transcript text with speaker labels in addition to metadata.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console logging level.",
    )
    return parser


def load_transcript_segments(path: Path) -> List[WhisperSegment]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "segments" in data:
        data = data["segments"]
    return [WhisperSegment(**segment) for segment in data]


def load_speaker_segments(path: Path) -> DiarizationResult:
    data = json.loads(path.read_text())
    segments = [SpeakerSegment(**segment) for segment in data.get("segments", [])]
    return DiarizationResult(segments=segments, num_speakers=data.get("num_speakers", 0))


def write_verbose_json(
    segments: List[WhisperSegment],
    output_path: Path,
    include_text: bool,
) -> None:
    text = " ".join(segment.text for segment in segments) if include_text else None
    response = TranscriptionResponse(
        text=text or "",
        segments=segments,
        language=None,
        model="parakeet-tdt-0.6b-v3",
    )
    payload = response.dict()
    if not include_text:
        payload["text"] = ""
    output_path.write_text(json.dumps(payload, indent=2))


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    segments_path = Path(args.segments).expanduser()
    speakers_path = Path(args.speakers).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transcript_segments = load_transcript_segments(segments_path)
    diarization_result = load_speaker_segments(speakers_path)

    logging.info(
        "Merging %d transcript segments with %d speaker segments",
        len(transcript_segments),
        len(diarization_result.segments),
    )

    merged = merge_diarization_with_transcription(
        diarization_result, transcript_segments
    )
    if args.include_speakers_in_text:
        apply_speaker_labels_to_text(merged)

    write_verbose_json(merged, output_path, include_text=args.include_speakers_in_text)
    logging.info("Merged output written to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

