import argparse
import json
import os
from pathlib import Path

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


def main():
    parser = argparse.ArgumentParser(description="Run Community-1 diarization.")
    parser.add_argument("audio_path", help="Path to 16kHz mono audio file.")
    parser.add_argument("output_json", help="Destination JSON file for segments.")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
        help="Hugging Face token (defaults to env var).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional speaker count hint.",
    )
    args = parser.parse_args()

    if not args.hf_token:
        raise SystemExit("Hugging Face token required via --hf-token or env var.")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=args.hf_token,
    )

    hook_kwargs = {}
    if args.num_speakers is not None:
        hook_kwargs["num_speakers"] = args.num_speakers

    with ProgressHook() as hook:
        output = pipeline(args.audio_path, hook=hook, **hook_kwargs)

    annotation = output.exclusive_speaker_diarization or output.speaker_diarization

    segments = [
        {
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
        }
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"segments": segments}, indent=2))
    print(f"Saved {len(segments)} segments to {out_path}")


if __name__ == "__main__":
    main()
