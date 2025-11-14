import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from diarization import (
    DiarizationResult,
    Diarizer,
    SpeakerSegment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run speaker diarization on an audio file and emit speaker segments."
    )
    parser.add_argument("--file", "-f", required=True, help="Path to the audio file.")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to write diarization JSON output.",
    )
    parser.add_argument(
        "--hf-token",
        help="Optional HuggingFace token (falls back to env/config).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Hint for the number of speakers (optional).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console logging level.",
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    audio_path = Path(args.file).expanduser()
    if not audio_path.exists():
        logging.error("Audio file %s does not exist.", audio_path)
        return 1

    token = args.hf_token or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    if not token:
        logging.error(
            "No HuggingFace token available. Set HUGGINGFACE_ACCESS_TOKEN or pass --hf-token."
        )
        return 1

    diarizer = Diarizer(access_token=token)
    result: DiarizationResult = diarizer.diarize(
        str(audio_path), num_speakers=args.num_speakers
    )

    output = {
        "audio_file": str(audio_path),
        "num_speakers": result.num_speakers,
        "segments": [segment.dict() for segment in result.segments],
    }

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    logging.info(
        "Saved diarization output for %d speakers to %s",
        result.num_speakers,
        output_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

