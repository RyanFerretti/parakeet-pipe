import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import yt_dlp


def sanitize_filename(name: str) -> str:
    """Create a filesystem-safe filename slug."""
    return re.sub(r"[^\w.\-]+", "_", name).strip("._") or "audio"


def resolve_output_path(
    url: str, audio_format: str, explicit_path: Optional[str]
) -> Path:
    if explicit_path:
        output_path = Path(explicit_path).expanduser()
        if not output_path.suffix:
            output_path = output_path.with_suffix(f".{audio_format}")
        return output_path

    metadata = {}
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            metadata = ydl.extract_info(url, download=False)
    except Exception:
        logging.warning("Unable to prefetch metadata for %s, using fallback name.", url)

    base_name = metadata.get("title") or metadata.get("id") or "audio"
    slug = sanitize_filename(base_name)
    downloads_dir = Path("downloads")
    downloads_dir.mkdir(parents=True, exist_ok=True)
    return downloads_dir / f"{slug}.{audio_format}"


def download_audio(
    url: str,
    output_path: Path,
    audio_format: str,
    quiet: bool,
    cookies_file: Optional[Path] = None,
) -> Path:
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template = str(output_path.parent / f"{output_path.stem}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": template,
        "noplaylist": True,
        "quiet": quiet,
        "no_warnings": quiet,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "192",
            }
        ],
        "postprocessor_args": [
            "-ar",
            "16000",
            "-ac",
            "1",
        ],
    }

    if cookies_file:
        resolved_cookies = Path(cookies_file).expanduser()
        if not resolved_cookies.exists():
            raise FileNotFoundError(f"Cookies file not found: {resolved_cookies}")
        ydl_opts["cookiefile"] = str(resolved_cookies)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    final_path = output_path.parent / f"{output_path.stem}.{audio_format}"
    if final_path != output_path:
        final_path.rename(output_path)

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video's audio track and save it locally."
    )
    parser.add_argument("--url", required=True, help="YouTube video URL.")
    parser.add_argument(
        "--output",
        "-o",
        help="Destination path for the audio file. Defaults to downloads/<title>.<ext>.",
    )
    parser.add_argument(
        "--audio-format",
        default="mp3",
        choices=["mp3", "wav", "flac", "m4a"],
        help="Audio format to extract via ffmpeg.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce yt-dlp logging noise.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console logging level.",
    )
    parser.add_argument(
        "--cookies-file",
        help="Path to a cookies.txt (Netscape format) file for authenticated YouTube downloads.",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_path = resolve_output_path(args.url, args.audio_format, args.output)
    logging.info("Downloading audio to %s", output_path)

    try:
        final_path = download_audio(
            args.url,
            output_path,
            args.audio_format,
            quiet=args.quiet,
            cookies_file=args.cookies_file,
        )
    except Exception as exc:
        logging.exception("Failed to download audio: %s", exc)
        return 1

    logging.info("Saved audio to %s", final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

