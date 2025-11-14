## YouTube → Transcript + Diarization Pipeline

This repository contains a standalone pipeline that:

1. downloads a YouTube video and resamples it to 16 kHz mono,
2. transcribes the audio with NVIDIA’s Parakeet-TDT ASR,
3. runs speaker diarization with pyannote’s Community‑1 pipeline, and
4. merges the speakers back into the transcript so each segment is labeled.

The legacy FastAPI server is **not** included—this repo is only the tooling required to run the four stages locally or in the cloud (e.g. Deepnote).

> **Why two virtual environments?**  
> Parakeet (via `nemo_toolkit`) and pyannote Community‑1 pin competing versions of `torch`, `torchaudio`, and `huggingface_hub`. Splitting them into `envs/parakeet` and `envs/community1` keeps the dependency trees isolated so you can install upgrades or patches without breaking the other stage.

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

Create the two envs up front:

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

> `requirements-community.txt` lists the minimal packages (`torch`, `torchaudio`, `pyannote.audio`). The extra index URL pulls NVIDIA’s CPU/GPU wheels; adjust to your platform as needed. Keeping Community‑1 slim avoids the heavy NeMo dependency graph.

Store your Hugging Face token once:

```bash
export HUGGINGFACE_ACCESS_TOKEN=hf_xxx
```

Add `.gitignore` entries for `envs/`, `downloads/`, `outputs/`, `*.wav`, `*.json`, etc. (see `.gitignore` in this repo).

---

### 2. Manual Stage-by-Stage Run

Assume `$ID` is the YouTube video ID. Each step writes into `downloads/` and `outputs/$ID/`.

```bash
# Download + resample (Parakeet env)
source envs/parakeet/bin/activate
python download_audio.py --url https://youtube.com/watch?v=$ID \
  --audio-format wav --output downloads/$ID.wav

# Transcribe with Parakeet (still in Parakeet env)
python cli.py --file downloads/$ID.wav \
  --format json --timestamps --disable-diarization \
  --segments-output outputs/${ID}/segments.json \
  --output outputs/${ID}/transcript.json

# Run Community‑1 diarization (activate diarizer env so pyannote deps stay separate)
source envs/community1/bin/activate
python community1.py downloads/$ID.wav outputs/${ID}/speakers.json

# Merge transcripts + speakers (switch back to Parakeet env for merge helpers)
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

`run_pipeline.py` drives the four stages automatically (download → ASR → diarization → merge) and writes every artifact into `outputs/<youtube_id>/`.

Run it from the Parakeet env and point it at the Community‑1 interpreter:

```bash
source envs/parakeet/bin/activate
python run_pipeline.py $ID \
  --hf-token "$HUGGINGFACE_ACCESS_TOKEN" \
  --community-python envs/community1/bin/python
```

Artifacts created per run:

```
downloads/<youtube_id>.wav        # normalized audio
outputs/<youtube_id>/transcript.json
outputs/<youtube_id>/segments.json
outputs/<youtube_id>/speakers.json
outputs/<youtube_id>/verbose.json
```

---

### 4. Deepnote / Cloud Workflow

Here’s a canonical setup for Deepnote (or any similar managed GPU notebook):

1. **Clone the repo** into a GPU-backed project (T4 or L4 is plenty).
2. **Create the two virtualenvs** exactly as described in Section 1. Example cell:
   ```bash
   python3.12 -m venv envs/parakeet
   source envs/parakeet/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Followed by a second cell for the Community‑1 env:
   ```bash
   python3.12 -m venv envs/community1
   source envs/community1/bin/activate
   pip install --upgrade pip
   pip install -r requirements-community.txt --extra-index-url https://download.pytorch.org/whl/cu121
   ```
3. **Store `HUGGINGFACE_ACCESS_TOKEN`** as a Deepnote secret or environment variable (Project Settings → Environment → Secrets).
4. **Run each stage from notebook cells**, making sure every cell activates the correct env:
   ```bash
   # Download
   source envs/parakeet/bin/activate && \
     python download_audio.py --url "https://youtube.com/watch?v=$ID" --audio-format wav --output downloads/$ID.wav

   # ASR
   source envs/parakeet/bin/activate && \
     python cli.py --file downloads/$ID.wav --format json --timestamps --disable-diarization \
       --segments-output outputs/$ID/segments.json --output outputs/$ID/transcript.json

   # Diarization
   source envs/community1/bin/activate && \
     python community1.py downloads/$ID.wav outputs/$ID/speakers.json

   # Merge
   source envs/parakeet/bin/activate && \
     python merge_segments.py \
       --segments outputs/$ID/segments.json \
       --speakers outputs/$ID/speakers.json \
       --include-speakers-in-text \
       --output outputs/$ID/verbose.json
   ```
   Or call the orchestrator instead:
   ```bash
   source envs/parakeet/bin/activate && \
     python run_pipeline.py $ID \
       --hf-token "$HUGGINGFACE_ACCESS_TOKEN" \
       --community-python envs/community1/bin/python
   ```
5. **Inspect the outputs** under `downloads/` and `outputs/<youtube_id>/` or push them to Deepnote artifacts.

**Timing:** Deepnote displays the runtime for each cell, so you can gauge per-stage latency directly. On a GPU runtime, ASR + diarization typically finish in under 10 minutes for an hour-long clip (versus ~35 minutes on CPU).

