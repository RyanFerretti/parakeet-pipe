## YouTube → Transcript + Diarization Pipeline

This repository contains a standalone pipeline that:

1. downloads a YouTube video and resamples it to 16 kHz mono,
2. transcribes the audio with NVIDIA’s Parakeet-TDT ASR,
3. runs speaker diarization with pyannote’s Community‑1 pipeline, and
4. merges the speakers back into the transcript so each segment is labeled.

The legacy FastAPI server is **not** included—this repo is only the tooling required to run the four stages locally or in the cloud (e.g. Deepnote).

---

### File Map

| File | Purpose |
| --- | --- |
| `download_audio.py` | Downloads/resamples a YouTube clip via `yt-dlp` + ffmpeg. |
| `cli.py` | Command-line interface for Parakeet transcription. Produces transcripts and optional `segments.json`. |
| `community1.py` | Minimal Community‑1 diarizer. Outputs `speakers.json` using pyannote.audio + ProgressHook. |
| `diarize_cli.py` | Scriptable entry point for diarization (same core logic as `community1.py`, but for batch jobs). |
| `merge_segments.py` | Merges Parakeet segments with Community‑1 speakers and emits `verbose.json`. |
| `run_pipeline.py` | Orchestrates all four stages end-to-end for a given YouTube ID or URL. |
| `audio.py`, `service.py`, `models.py`, `transcription.py`, `diarization/` | Shared logic used by the CLI (conversion, chunking, pydantic models, pyannote helpers). |
| `requirements.txt` | Dependency set for the Parakeet/CLI environment. |
| `requirements-community.txt` | Minimal deps for the Community-1 diarizer env. |

---

### 1. Environments (Python 3.12)

Create two virtualenvs so Parakeet (NeMo) and Community‑1 can pin their own dependencies:

```bash
python3.12 -m venv envs/parakeet
source envs/parakeet/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3.12 -m venv envs/community1
source envs/community1/bin/activate
pip install --upgrade pip
pip install -r requirements-community.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

> `requirements-community.txt` lists the minimal packages (`torch`, `torchaudio`, `pyannote.audio`). The extra index URL pulls NVIDIA’s CPU/GPU wheels; adjust to your platform as needed.

Store your Hugging Face token once:

```bash
export HUGGINGFACE_ACCESS_TOKEN=hf_xxx
```

Add `.gitignore` entries for `envs/`, `downloads/`, `outputs/`, `*.wav`, `*.json`, etc. (see `.gitignore` in this repo).

---

### 2. Manual Stage-by-Stage Run

Assume `$ID` is the YouTube video ID.

```bash
# Download + resample
source envs/parakeet/bin/activate
python download_audio.py --url https://youtube.com/watch?v=$ID \
  --audio-format wav --output downloads/$ID.wav

# Transcribe with Parakeet (creates transcript + segments)
python cli.py --file downloads/$ID.wav \
  --format json --timestamps --disable-diarization \
  --segments-output outputs/${ID}/segments.json \
  --output outputs/${ID}/transcript.json

# Run Community‑1 diarization (use the slim env)
source envs/community1/bin/activate
python community1.py downloads/$ID.wav outputs/${ID}/speakers.json

# Merge transcripts + speakers
source envs/parakeet/bin/activate
python merge_segments.py \
  --segments outputs/${ID}/segments.json \
  --speakers outputs/${ID}/speakers.json \
  --include-speakers-in-text \
  --output outputs/${ID}/verbose.json
```

`outputs/${ID}/verbose.json` matches OpenAI Whisper’s `verbose_json` format but includes speaker labels.

---

### 3. One-Command Orchestration

`run_pipeline.py` drives the four steps automatically. Run it from the Parakeet env and point it at the Community‑1 interpreter:

```bash
source envs/parakeet/bin/activate
python run_pipeline.py $ID \
  --hf-token "$HUGGINGFACE_ACCESS_TOKEN" \
  --community-python envs/community1/bin/python
```

Artifacts land in `downloads/` and `outputs/<youtube_id>/`.

---

### 4. Deepnote / Cloud Workflow

1. Clone this repo into a GPU-backed workspace (T4/L4 are sufficient).
2. Recreate the two virtualenvs (commands above).
3. Store `HUGGINGFACE_ACCESS_TOKEN` as a secret/env variable.
4. Run `run_pipeline.py` or add a controller notebook with cells like:

```python
!source envs/parakeet/bin/activate && python download_audio.py --url ...
!source envs/parakeet/bin/activate && python cli.py ...
!source envs/community1/bin/activate && python community1.py ...
!source envs/parakeet/bin/activate && python merge_segments.py ...
```

On a GPU runtime, ASR + diarization finish in <10 minutes for a 60‑minute clip, versus ~35 minutes on CPU.

