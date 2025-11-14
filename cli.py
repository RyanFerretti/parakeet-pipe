import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from config import get_config
from service import TranscriptionService
from transcription import format_srt, format_vtt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Parakeet transcription + diarization pipeline from the command line."
    )
    parser.add_argument(
        "--file",
        "-f",
        required=True,
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument(
        "--format",
        "-F",
        default="json",
        choices=["json", "text", "srt", "vtt", "verbose_json"],
        help="Response format to emit.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to write the output. Defaults to stdout.",
    )
    parser.add_argument(
        "--segments-output",
        help="Optional path to dump raw transcription segments as JSON (for pipeline use).",
    )
    parser.add_argument(
        "--language",
        "-l",
        help="Optional language hint passed to the ASR model.",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include segment timestamps in the JSON response.",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Request word-level timestamps from the model.",
    )
    parser.add_argument(
        "--hf-token",
        help="Override the HuggingFace token for diarization (falls back to env/config).",
    )
    parser.add_argument(
        "--temp-dir",
        help="Override temporary working directory for audio artifacts.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity.",
    )

    parser.add_argument(
        "--enable-diarization",
        dest="diarize",
        action="store_true",
        help="Force-enable speaker diarization.",
    )
    parser.add_argument(
        "--disable-diarization",
        dest="diarize",
        action="store_false",
        help="Disable speaker diarization.",
    )
    parser.add_argument(
        "--include-speakers-in-text",
        dest="include_diarization_in_text",
        action="store_true",
        help="Prefix transcript text with speaker labels.",
    )
    parser.add_argument(
        "--exclude-speakers-from-text",
        dest="include_diarization_in_text",
        action="store_false",
        help="Omit speaker labels from transcript text.",
    )

    parser.set_defaults(diarize=None, include_diarization_in_text=None)

    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    audio_path = Path(args.file).expanduser()
    if not audio_path.exists():
        logging.error("Audio file %s does not exist.", audio_path)
        return 1

    config = get_config()
    if args.hf_token:
        config.update_hf_token(args.hf_token)
    if args.temp_dir:
        config.temp_dir = args.temp_dir
        Path(config.temp_dir).mkdir(parents=True, exist_ok=True)

    if config.get_hf_token():
        logging.info("HuggingFace token detected; diarization is available.")
    else:
        logging.warning("No HuggingFace token detected; diarization requests will be skipped.")

    service = TranscriptionService(config=config)

    needs_segments = args.timestamps or args.format in {"verbose_json", "srt", "vtt"}

    try:
        response = service.transcribe_file(
            str(audio_path),
            language=args.language,
            diarize=args.diarize,
            include_diarization_in_text=args.include_diarization_in_text,
            word_timestamps=args.word_timestamps,
            return_segments=needs_segments,
        )
    except Exception as exc:
        logging.exception("Transcription failed: %s", exc)
        return 1

    output = render_output(response, args.format)

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.write_text(output)
        logging.info("Wrote %s output to %s", args.format, output_path)
    else:
        sys.stdout.write(output)
        if not output.endswith("\n"):
            sys.stdout.write("\n")

    if args.segments_output:
        if not response.segments:
            logging.warning("Segments unavailable; skipping segments-output write.")
        else:
            segments_payload = [segment.dict() for segment in response.segments]
            segments_path = Path(args.segments_output).expanduser()
            segments_path.parent.mkdir(parents=True, exist_ok=True)
            segments_path.write_text(json.dumps(segments_payload, indent=2))
            logging.info("Saved %d segments to %s", len(response.segments), segments_path)

    return 0


def render_output(response, response_format: str) -> str:
    if response_format in {"json", "verbose_json"}:
        return json.dumps(response.dict(), indent=2)
    if response_format == "text":
        return response.text or ""
    if response_format == "srt":
        if not response.segments:
            raise RuntimeError("Segments not available for SRT output")
        return format_srt(response.segments)
    if response_format == "vtt":
        if not response.segments:
            raise RuntimeError("Segments not available for VTT output")
        return format_vtt(response.segments)
    raise ValueError(f"Unsupported response format: {response_format}")


if __name__ == "__main__":
    sys.exit(main())

