import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd, env=None, cwd=None, label=""):
    display = " ".join(cmd)
    print(f"\n{'-' * 80}\nRunning {label or cmd[0]}:\n{display}\n{'-' * 80}")
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def resolve_youtube_url(raw: str) -> str:
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    return f"https://www.youtube.com/watch?v={raw}"


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: download → ASR → diarize → merge."
    )
    parser.add_argument("youtube", help="YouTube URL or ID")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
        help="Hugging Face token (falls back to env var)",
    )
    parser.add_argument(
        "--community-python",
        default=str(Path("community1-venv/bin/python")),
        help="Python interpreter for the Community-1 diarizer",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to store transcripts and merged output",
    )
    parser.add_argument(
        "--audio-dir",
        default="downloads",
        help="Directory to store downloaded audio",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional speaker count hint for diarization",
    )
    parser.add_argument(
        "--download-cookies-file",
        help="Optional cookies.txt file passed to download_audio.py for authenticated YouTube downloads.",
    )
    args = parser.parse_args()

    if not args.hf_token:
        raise SystemExit("Hugging Face token required via --hf-token or env var.")

    youtube_url = resolve_youtube_url(args.youtube)
    youtube_id = youtube_url.split("v=")[-1]

    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{youtube_id}.wav"

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    job_output_dir = output_root / youtube_id
    job_output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = job_output_dir / "transcript.json"
    segments_path = job_output_dir / "segments.json"
    speakers_path = job_output_dir / "speakers.json"
    merged_path = job_output_dir / "verbose.json"

    env_with_token = os.environ.copy()
    env_with_token["HUGGINGFACE_ACCESS_TOKEN"] = args.hf_token

    # 1. Download + resample
    download_cmd = [
        sys.executable,
        "download_audio.py",
        "--url",
        youtube_url,
        "--audio-format",
        "wav",
        "--output",
        str(audio_path),
    ]
    if args.download_cookies_file:
        download_cmd.extend(["--cookies-file", args.download_cookies_file])

    # 🔹 Tiny guard: skip download if audio already exists
    if audio_path.exists():
        print(f"Skipping download; found existing {audio_path}")
    else:
        run(download_cmd, label="Download audio")

    # 2. Parakeet ASR
    run(
        [
            sys.executable,
            "cli.py",
            "--file",
            str(audio_path),
            "--format",
            "json",
            "--timestamps",
            "--disable-diarization",
            "--segments-output",
            str(segments_path),
            "--output",
            str(transcript_path),
        ],
        env=env_with_token,
        label="Parakeet transcription",
    )

    # 3. Community-1 diarization
    community_cmd = [
        args.community_python,
        "community1.py",
        str(audio_path),
        str(speakers_path),
    ]
    if args.num_speakers is not None:
        community_cmd.extend(["--num-speakers", str(args.num_speakers)])
    run(
        community_cmd,
        env=env_with_token,
        label="Community-1 diarization",
    )

    # 4. Merge
    run(
        [
            sys.executable,
            "merge_segments.py",
            "--segments",
            str(segments_path),
            "--speakers",
            str(speakers_path),
            "--include-speakers-in-text",
            "--output",
            str(merged_path),
        ],
        label="Merge ASR + diarization",
    )

    print(
        f"\nWorkflow finished. Files:\n"
        f"  Audio:        {audio_path}\n"
        f"  Transcript:   {transcript_path}\n"
        f"  Segments:     {segments_path}\n"
        f"  Speakers:     {speakers_path}\n"
        f"  Merged (JSON): {merged_path}\n"
    )


if __name__ == "__main__":
    main()

