from __future__ import annotations
from difflib import SequenceMatcher
import os
import tempfile
import threading
import time
import logging
from collections import Counter, deque
from datetime import datetime
from fractions import Fraction
from typing import Deque, Dict, List, Optional, Tuple
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import pyaudiowpatch as pyaudio
import scipy.io.wavfile as wav_io
import scipy.signal as signal

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("transcriber.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("live_transcriber")

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000          # Whisper native — do NOT change

WINDOW_SECONDS: int = 6           # Rolling context window fed to the model
STEP_SECONDS: int = 3              # Capture + inference cadence

TRANSCRIPT_FILE: str   = "transcript1.txt"
MIC_AUDIO_FILE: str    = "recorded_mic1.wav"
SPEAKER_AUDIO_FILE: str = "recorded_speaker1.wav"

# ── Silence gates ─────────────────────────────────────────────
# YOU  stream: RMS must exceed this to trigger transcription
SILENCE_THRESHOLD: float = 0.006
# CLIENT stream multiplier.
# FIX: was 2.5 — too aggressive, dropped soft-spoken Hindi callers.
# 1.5 keeps the gate meaningful while passing normal call-centre audio.
CLIENT_SILENCE_MUL: float = 1.5

# ── Pipeline chunk config (Hinglish model) ───────────────────
PIPE_CHUNK_LENGTH_S: int  = 15
PIPE_STRIDE_LENGTH_S: int = 2

HINGLISH_MODEL_ID: str = "Oriserve/Whisper-Hindi2Hinglish-Swift"

# Initial prompt injected into the Whisper decoder.
# This single line shifts the model's prior strongly toward
# Hinglish output instead of pure Devanagari Hindi or pure English.
HINGLISH_INITIAL_PROMPT: str = (
    "Yeh ek Hinglish conversation hai. "
    "Roman script mein transcribe karo. "
    "Hindi aur English mix hogi."
)

# ─────────────────────────────────────────────────────────────
#  GLOBALS
# ─────────────────────────────────────────────────────────────

processor = None

mic_rate_global: int        = 44_100
speaker_rate_global: int    = 48_000
selected_language_code: Optional[str] = None
selected_language_name: Optional[str] = None
use_indic_model: bool       = False
fw_model                    = None   # faster-whisper WhisperModel (English)
indic_pipe                  = None   # HuggingFace pipeline (Hinglish)
_last_speaker: str = ""
_pending_lines: Dict[str, str] = {"YOU": "", "CLIENT": ""}

# ── Lock-free audio buffers (deque.extend is GIL-atomic in CPython) ──
mic_buffer_ts: deque     = deque()
speaker_buffer_ts: deque = deque()
mic_full: deque          = deque()
speaker_full: deque      = deque()

mic_window: Optional[Deque]     = None
speaker_window: Optional[Deque] = None

# ── Latest-snapshot slots (one per stream) ───────────────────
_snap_lock = threading.Lock()
_mic_snap: Dict     = {"data": None, "rate": 0, "ready": False}
_speaker_snap: Dict = {"data": None, "rate": 0, "ready": False}

_drop_counts: Dict[str, int] = {"YOU": 0, "CLIENT": 0}
from collections import deque as _deque
_recent_text: Dict[str, _deque] = {
    "YOU":    _deque(maxlen=10),
    "CLIENT": _deque(maxlen=10),
}


# ─────────────────────────────────────────────────────────────
#  LANGUAGE SELECTION
# ─────────────────────────────────────────────────────────────
def select_language() -> Tuple[str, str, bool]:
    """Prompt the user to choose a transcription language/mode."""
    print("\n  LANGUAGE / TRANSCRIPT MODE")
    print("  [1]  Hinglish  (Hindi + English mixed — uses Oriserve/Whisper-Hindi2Hinglish-Swift)")
    print("  [2]  English only  (uses faster-whisper small / int8)\n")
    while True:
        try:
            choice = int(input("  Enter your choice (1 / 2): ").strip())
            if choice == 1:
                return "hi", "Hinglish", True
            if choice == 2:
                return "en", "English", False
            print("  Please enter 1 or 2.")
        except ValueError:
            print("  Invalid input — numbers only.")


# ─────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────
def load_models(use_indic: bool) -> None:
    """Load the appropriate ASR model based on language selection."""
    global fw_model, indic_pipe
    global processor, prompt_ids
    if use_indic:
        log.info(f"Loading {HINGLISH_MODEL_ID} for Hinglish transcription ...")
        import torch
        import warnings
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )
        warnings.filterwarnings("ignore")

        # Use float16 on CUDA if available for 2× speed; fall back to float32
        device      = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        log.info(f"  Device: {device}  dtype: {torch_dtype}")

        hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            HINGLISH_MODEL_ID,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        hf_model.to(device)

        processor = AutoProcessor.from_pretrained(HINGLISH_MODEL_ID)

        # Pipeline with chunked long-form inference and stride overlap so that
        # words at chunk boundaries are not lost.
        indic_pipe = pipeline(
            "automatic-speech-recognition",
            model=hf_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=PIPE_CHUNK_LENGTH_S,
            stride_length_s=PIPE_STRIDE_LENGTH_S,
            return_timestamps=False,
        )
        log.info(f"{HINGLISH_MODEL_ID} ready ✓")

    else:
        log.info("Loading faster-whisper (small, int8) for English ...")
        from faster_whisper import WhisperModel
        fw_model = WhisperModel("small", device="cpu", compute_type="int8")
        log.info("faster-whisper ready ✓")


# ─────────────────────────────────────────────────────────────
#  DEVICE SELECTION
# ─────────────────────────────────────────────────────────────
def select_microphone(p: pyaudio.PyAudio) -> Tuple[int, int]:
    """List input devices and let the user pick a microphone."""
    print("\n  AVAILABLE MICROPHONES")
    mic_devices: List[dict] = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and not info.get("isLoopbackDevice", False):
            if "stereo mix" not in info["name"].lower():
                mic_devices.append({
                    "index":       i,
                    "name":        info["name"],
                    "channels":    info["maxInputChannels"],
                    "sample_rate": int(info["defaultSampleRate"]),
                })
    if not mic_devices:
        raise RuntimeError("No microphone found!")
    for idx, d in enumerate(mic_devices):
        print(f"  [{idx}] {d['name']}  ({d['channels']}ch @ {d['sample_rate']} Hz)")
    while True:
        try:
            c = int(input(f"\n  Select microphone (0-{len(mic_devices)-1}): "))
            if 0 <= c < len(mic_devices):
                sel = mic_devices[c]
                log.info(f"Mic selected: {sel['name']}")
                return sel["index"], sel["sample_rate"]
            print(f"  Enter 0-{len(mic_devices)-1}")
        except ValueError:
            print("  Invalid input.")


def select_speaker(p: pyaudio.PyAudio) -> Tuple[Optional[int], Optional[int]]:
    """List loopback/stereo-mix devices and let the user pick one."""
    print("\n  AVAILABLE SPEAKER CAPTURE DEVICES")
    spk_devices: List[dict] = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            name = info["name"].lower()
            if "stereo mix" in name or info.get("isLoopbackDevice", False):
                spk_devices.append({
                    "index":       i,
                    "name":        info["name"],
                    "channels":    info["maxInputChannels"],
                    "sample_rate": int(info["defaultSampleRate"]),
                })
    if not spk_devices:
        log.warning("No speaker capture device — speaker stream disabled.")
        return None, None
    for idx, d in enumerate(spk_devices):
        print(f"  [{idx}] {d['name']}  ({d['channels']}ch @ {d['sample_rate']} Hz)")
    print("=" * 54)
    while True:
        try:
            c = int(input(f"  Select speaker capture (0-{len(spk_devices)-1}): "))
            if 0 <= c < len(spk_devices):
                sel = spk_devices[c]
                log.info(f"Speaker selected: {sel['name']}")
                return sel["index"], sel["sample_rate"]
            print(f"  Enter 0-{len(spk_devices)-1}")
        except ValueError:
            print("  Invalid input.")


# ─────────────────────────────────────────────────────────────
#  AUDIO CALLBACKS  — lock-free (GIL-atomic deque ops)
# ─────────────────────────────────────────────────────────────
def mic_callback(in_data, frame_count, time_info, status) -> tuple:
    audio = np.frombuffer(in_data, dtype=np.float32)
    mic_buffer_ts.extend(audio.tolist())
    mic_full.extend(audio.tolist())
    return (None, pyaudio.paContinue)


def speaker_callback(in_data, frame_count, time_info, status) -> tuple:
    audio = np.frombuffer(in_data, dtype=np.float32)
    # Speaker loopback arrives as 2-channel interleaved [L, R, L, R, ...].
    # Average to mono here so downstream code always sees 1-D audio at the
    # correct sample count.  Skipping this step would cause audio to be
    # treated as mono at 2× the actual sample rate (double-speed playback).
    if audio.size % 2 == 0:
        audio = audio.reshape(-1, 2).mean(axis=1)
    speaker_buffer_ts.extend(audio.tolist())
    speaker_full.extend(audio.tolist())
    return (None, pyaudio.paContinue)


# ─────────────────────────────────────────────────────────────
#  DRAIN HELPER
# ─────────────────────────────────────────────────────────────
def _drain(buf: deque) -> list:
    """Atomically drain all items from a deque into a list."""
    out: list = []
    try:
        while True:
            out.append(buf.popleft())
    except IndexError:
        pass
    return out


# ─────────────────────────────────────────────────────────────
#  AUDIO HELPERS
# ─────────────────────────────────────────────────────────────
def _resample(a: np.ndarray, src: int, dst: int = SAMPLE_RATE) -> np.ndarray:
    """Polyphase resample using FIR filter (O(n), avoids FFT edge ringing)."""
    if src == dst:
        return a
    frac = Fraction(dst, src).limit_denominator(1000)
    return signal.resample_poly(a, frac.numerator, frac.denominator)


def _prepare(audio_data: list, source_rate: int) -> np.ndarray:
    """Convert raw callback samples → normalised mono float32 at SAMPLE_RATE."""
    a = np.array(audio_data, dtype=np.float32)
    if a.ndim == 2:          # defensive: should already be 1-D
        a = a.mean(axis=1)
    return _resample(a, source_rate, SAMPLE_RATE)


# ─────────────────────────────────────────────────────────────
#  HALLUCINATION FILTER
# ─────────────────────────────────────────────────────────────
# Common outputs the models produce when there is no real speech or when
# the model is guessing.  Keep this list lean — over-filtering hurts recall.
_KNOWN_HALLUCINATIONS: frozenset = frozenset({
    "thanks for watching", "thank you for watching", "subscribe",
    "please subscribe", "subtitles by", "dhanyavaad", "shukriya",
    "thank you", "bye bye", ".",
    # Repetitive Hinglish filler strings  (single occurrences are valid speech)
    "haan haan", "haan haan haan", "haan haan haan haan",
    "achchha achchha", "achchha achchha achchha",
    "ji ji", "nahin nahin",
})


def _is_hallucination(text: str) -> bool:
    """
    Return True if the transcription looks like model confabulation.

    Design principles
    ─────────────────
    • Single short tokens like "haan", "ji", "ok" are NOT filtered here —
      they are legitimate Hinglish acknowledgements.
    • The uniqueness filter is mode-gated: English short phrases have many
      valid 1-2-unique-word utterances ("I see", "yes please").
    • Repetition checks require ≥6 tokens to avoid false positives on
      short genuine utterances.
    """
    t = text.strip()

    # Absolute minimum length — catches stray punctuation / single chars
    # FIX: was 5; lowered to 3 so "ok", "ha", "no" pass through.
    if len(t) < 3:
        return True

    if t.lower().rstrip(".") in _KNOWN_HALLUCINATIONS:
        return True

    tokens = t.split()

    # All tokens identical (e.g. "yes yes yes yes")
    if len(tokens) >= 2 and len({tok.lower() for tok in tokens}) == 1:
        return True

    # Short phrase with ≤2 unique words — only block in Hinglish mode
    # because English frequently produces valid 2-unique-word phrases.
    if use_indic_model and len(tokens) <= 5 and len({tok.lower() for tok in tokens}) <= 2:
        return True

    if len(tokens) >= 6:
        # Any single token dominating >35% of the utterance → repetition loop
        most_common_count = Counter(tok.lower() for tok in tokens).most_common(1)[0][1]
        if most_common_count / len(tokens) > 0.35:
            return True
        # Trailing trigram repeated ≥2 times elsewhere in the utterance
        tail = tokens[-3:]
        repeats = sum(
            1 for i in range(len(tokens) - 3) if tokens[i : i + 3] == tail
        )
        if repeats >= 2:
            return True

    return False


# ─────────────────────────────────────────────────────────────
#  DEDUPLICATION
# ─────────────────────────────────────────────────────────────
# Require at least 3 words of overlap before stripping the prefix.
# This prevents the deduplicator eating the start of new sentences that
# coincidentally share a common word with the previous output.
MIN_DEDUP_OVERLAP: int = 3

def _deduplicate(new_text: str, last_text: str) -> str:
    """
    Strip leading words from new_text that already appeared at the end of
    last_text. Uses fuzzy matching to handle slight model variance between
    overlapping windows.
    """
    if not last_text:
        return new_text

    new_words  = new_text.split()
    last_words = last_text.split()
    max_overlap = min(len(last_words), len(new_words))

    best = 0
    for n in range(max_overlap, MIN_DEDUP_OVERLAP - 1, -1):
        last_tail = " ".join(last_words[-n:])
        new_head  = " ".join(new_words[:n])
        ratio = SequenceMatcher(None, last_tail.lower(), new_head.lower()).ratio()
        if ratio >= 0.75:   # 75% similarity counts as overlap
            best = n
            break

    if best < MIN_DEDUP_OVERLAP:
        return new_text

    deduplicated = " ".join(new_words[best:]).strip()
    return deduplicated if deduplicated else new_text

# ─────────────────────────────────────────────────────────────
#  LOW-CONFIDENCE FILTER
# ─────────────────────────────────────────────────────────────
# Common Hinglish fillers.  These are legitimate in isolation but suspicious
# when they make up the majority of a short utterance (model guessing).
_HINGLISH_FILLERS: frozenset = frozenset({
    "haan", "ji", "aur", "yah", "vah", "hai", "toh",
    "achchha", "nahin", "ek", "ki", "ka", "ke", "ko",
})


def _is_low_confidence(text: str) -> bool:
    """
    Catch confabulation patterns produced when the model guesses.

    Changes from previous version
    ──────────────────────────────
    • Hinglish filler ratio threshold: 0.70 → 0.85
      At 0.70 valid phrases like "haan ji theek hai" (4 tokens, 2 fillers
      = 0.50 ratio) passed, but "haan ji" (2 tokens, 2 fillers = 1.0 ratio)
      was caught.  The old threshold of 0.70 was catching too many edge cases.
      0.85 is more conservative.
    • Minimum length check uses characters not tokens for Hinglish so that
      2-char Roman tokens ("ji", "ok", "ha") are allowed.
    """
    if not text or len(text.strip()) < 2:
        return True

    tokens = text.strip().split()
    if len(tokens) < 1:
        return True

    if use_indic_model:
        # FIX: threshold raised from 0.70 → 0.85 to stop valid fillers being
        # discarded when they constitute a high proportion of short utterances.
        filler_count = sum(
            1 for t in tokens if t.lower().rstrip(".,") in _HINGLISH_FILLERS
        )
        if len(tokens) <= 6 and filler_count / len(tokens) > 0.85:
            return True

    # Average word length < 2.5 chars → model outputting noise / single chars
    avg_len = sum(len(t) for t in tokens) / len(tokens)
    if avg_len < 2.5:
        return True

    lower_tokens = [t.lower().rstrip(".,") for t in tokens]
    unique_count = len(set(lower_tokens))
    if unique_count <= 2 and len(tokens) >= 2:
        if not use_indic_model and len(tokens) <= 3:
            pass   # English short phrases like "yes please" are valid
        else:
            return True

    return False


# ─────────────────────────────────────────────────────────────
#  EMIT / PRINT HELPERS
# ─────────────────────────────────────────────────────────────
def _emit(label: str, msg: str) -> None:
    """Buffer a debug/status line; flush previous speaker's pending line on speaker change."""
    global _last_speaker
    if _last_speaker and _last_speaker != label:
        pending = _pending_lines.get(_last_speaker, "")
        if pending:
            print(pending, flush=True)
            _pending_lines[_last_speaker] = ""
        print()
    _last_speaker = label
    _pending_lines[label] = f"[{label}] {msg}"


# ─────────────────────────────────────────────────────────────
#  TRANSCRIPTION  — core function
# ─────────────────────────────────────────────────────────────
def transcribe_audio(audio_data: list, source_label: str, source_rate: int) -> None:
    """
    Run ASR on one audio snapshot and write the result to the transcript.

    Parameters
    ──────────
    audio_data   : raw float32 samples at source_rate
    source_label : "YOU" (microphone) or "CLIENT" (speaker loopback)
    source_rate  : original capture sample rate (before resampling)
    """
    if not audio_data:
        _emit(source_label, "no audio data — skipping")
        return

    audio_np = _prepare(audio_data, source_rate)
    raw_max  = float(np.max(np.abs(audio_np)))
    rms      = float(np.sqrt(np.mean(audio_np ** 2)))

    gate = (
        SILENCE_THRESHOLD if source_label == "YOU"
        else SILENCE_THRESHOLD * CLIENT_SILENCE_MUL
    )

    if rms < gate:
        _emit(source_label, f"silent  rms={rms:.4f}  gate={gate:.4f} — skipping")
        return

    _emit(source_label, f"transcribing…  rms={rms:.4f}  samples={len(audio_np):,}")

    # Peak-normalise so soft speech reaches the model at a consistent level
    if raw_max > 0:
        audio_np = audio_np / raw_max

    chunk_text = ""

    # ── HINGLISH path ────────────────────────────────────────
    if use_indic_model and indic_pipe is not None:
        try:
            prompt = HINGLISH_INITIAL_PROMPT
            import torch

            prompt_ids = torch.tensor(
                processor.tokenizer(HINGLISH_INITIAL_PROMPT).input_ids,
                dtype=torch.long
            )
            result = indic_pipe(
                audio_np.copy(),
            )
            raw_text = result.get("text", "").strip()
            log.debug(f"[{source_label}] RAW: {raw_text[:80]!r}")

            if _is_low_confidence(raw_text):
                log.debug(
                    f"[{source_label}] low-confidence — rejecting: {raw_text[:60]!r}"
                )
                return

            chunk_text = raw_text
            _emit(source_label, f"model returned: {chunk_text[:80]!r}")

        except Exception as exc:
            log.error(f"[{source_label}] Hinglish model error: {exc}", exc_info=True)
            return

    # ── ENGLISH path (faster-whisper) ────────────────────────
    else:
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                wav_io.write(tmp_path, SAMPLE_RATE, (audio_np * 32767).astype(np.int16))

            segments, _ = fw_model.transcribe(
                tmp_path,
                language=selected_language_code,
                task="transcribe",
                beam_size=5,
                vad_filter=False,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=600,
                    threshold=0.2,
                ),
                # Previous-text conditioning causes error compounding in live
                # sliding-window mode — disable for live use.
                condition_on_previous_text=False,
                # 0.35 is correct: 0.85 (old) kept segments even when Whisper
                # was 85% confident there was NO speech.
                no_speech_threshold=0.25,
                repetition_penalty=1.3,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
            )

            parts: List[str] = []
            for seg in segments:
                if seg.no_speech_prob <= 0.35:
                    _emit(source_label, f"segment kept: {seg.text.strip()[:60]!r}")
                    parts.append(seg.text.strip())
                else:
                    _emit(
                        source_label,
                        f"segment dropped (no_speech={seg.no_speech_prob:.2f}): "
                        f"{seg.text.strip()[:50]!r}",
                    )

            chunk_text = " ".join(parts)

        except Exception as exc:
            log.error(f"[{source_label}] faster-whisper error: {exc}", exc_info=True)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ── Post-processing (common to both paths) ────────────────
    if not chunk_text:
        _emit(source_label, "empty result after transcription — skipping")
        return

    if _is_hallucination(chunk_text):
        _emit(source_label, f"hallucination filtered: {chunk_text[:60]!r}")
        return

    last_combined = " ".join(_recent_text[source_label])
    clean_text = _deduplicate(chunk_text, last_combined)
    if not clean_text:
        _emit(source_label, "fully duplicate — nothing new to emit")
        return

    if _is_hallucination(clean_text):
        _emit(source_label, "deduplicated text is hallucination — skipping")
        return

    _recent_text[source_label].append(chunk_text)  # store pre-dedup for next overlap check

    # Flush any pending status line before printing the transcript line
    pending = _pending_lines.get(source_label, "")
    if pending:
        print(pending, flush=True)
        _pending_lines[source_label] = ""

    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{source_label}] [{selected_language_name}] {clean_text}"
    print(f"  >>> {line}", flush=True)
    print()
    log.info(f"  {line}")
    with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ─────────────────────────────────────────────────────────────
#  SNAPSHOT HELPERS
# ─────────────────────────────────────────────────────────────
def _put_snapshot(snap: dict, audio: list, rate: int, label: str) -> None:
    """Write a new audio snapshot; log if previous snapshot was not consumed."""
    if not audio:
        return
    with _snap_lock:
        if snap["ready"]:
            _drop_counts[label] = _drop_counts.get(label, 0) + 1
            log.debug(
                f"[{label}] Stale snapshot overwritten "
                f"(skipped inferences this session: {_drop_counts[label]})"
            )
        snap["data"]  = audio
        snap["rate"]  = rate
        snap["ready"] = True


def _get_snapshot(snap: dict) -> Tuple[Optional[list], int]:
    """Atomically consume and return a snapshot (returns None if no new data)."""
    with _snap_lock:
        if not snap["ready"]:
            return None, 0
        audio       = snap["data"]
        rate        = snap["rate"]
        snap["data"]  = None
        snap["ready"] = False
    return audio, rate


# ─────────────────────────────────────────────────────────────
#  WORKER FACTORY
# ─────────────────────────────────────────────────────────────
def _make_worker(
    snap: dict,
    stop_event: threading.Event,
    label: str,
    name: str,
) -> threading.Thread:
    """
    Spawn a daemon thread that consumes snapshots and calls transcribe_audio.
    The thread drains one final snapshot after the stop event fires so that
    the last utterance before Ctrl+C is not lost.
    """
    def _run() -> None:
        log.info(f"Worker '{name}' started")
        while not stop_event.is_set():
            audio, rate = _get_snapshot(snap)
            if audio is not None:
                try:
                    transcribe_audio(audio, label, rate)
                except Exception as exc:
                    log.error(f"[{label}] worker unhandled exception: {exc}", exc_info=True)
            else:
                time.sleep(0.1)
        # Drain any remaining snapshot after shutdown signal
        audio, rate = _get_snapshot(snap)
        if audio is not None:
            try:
                transcribe_audio(audio, label, rate)
            except Exception as exc:
                log.error(f"[{label}] shutdown drain error: {exc}", exc_info=True)
        log.info(f"Worker '{name}' finished")

    t = threading.Thread(target=_run, daemon=True, name=name)
    t.start()
    return t


# ─────────────────────────────────────────────────────────────
#  SAVE FULL AUDIO
# ─────────────────────────────────────────────────────────────
def save_full_audio() -> None:
    """
    Write the complete session audio to WAV files at SAMPLE_RATE (16 kHz).
    Both streams are resampled before writing so the saved files are
    directly playable without further conversion.
    """
    mic_data = _drain(mic_full)
    spk_data = _drain(speaker_full)

    if mic_data:
        a = _prepare(mic_data, mic_rate_global)
        wav_io.write(MIC_AUDIO_FILE, SAMPLE_RATE, (a * 32767).astype(np.int16))
        log.info(f"Mic     -> {MIC_AUDIO_FILE}  ({len(a)/SAMPLE_RATE:.1f}s @ {SAMPLE_RATE} Hz)")

    if spk_data:
        a = _prepare(spk_data, speaker_rate_global)
        wav_io.write(SPEAKER_AUDIO_FILE, SAMPLE_RATE, (a * 32767).astype(np.int16))
        log.info(f"Speaker -> {SPEAKER_AUDIO_FILE}  ({len(a)/SAMPLE_RATE:.1f}s @ {SAMPLE_RATE} Hz)")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main() -> None:
    global mic_rate_global, speaker_rate_global
    global selected_language_code, selected_language_name, use_indic_model
    global mic_window, speaker_window

    # 1. Language / mode selection
    lang_code, lang_name, use_indic = select_language()
    selected_language_code = lang_code
    selected_language_name = lang_name
    use_indic_model        = use_indic

    # 2. Audio device selection
    p = pyaudio.PyAudio()
    mic_index, mic_rate = select_microphone(p)
    spk_index, spk_rate = select_speaker(p)
    mic_rate_global = mic_rate
    if spk_rate:
        speaker_rate_global = spk_rate

    # 3. Load ASR model
    load_models(use_indic)

    # 4. Initialise transcript file
    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as fh:
        fh.write(f"=== Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        fh.write(f"=== Language: {lang_name} ===\n\n")

    # 5. Initialise rolling context windows
    mic_window     = deque(maxlen=int(WINDOW_SECONDS * mic_rate_global))
    speaker_window = deque(maxlen=int(WINDOW_SECONDS * speaker_rate_global))

    log.info("-" * 58)
    log.info(f"Transcript  -> {TRANSCRIPT_FILE}")
    log.info(f"Language    -> {lang_name}")
    log.info(f"Window      -> {WINDOW_SECONDS}s context / {STEP_SECONDS}s step")
    log.info(f"Silence MUL -> CLIENT×{CLIENT_SILENCE_MUL}  YOU×1.0")
    log.info("Press Ctrl+C to stop")
    log.info("-" * 58)

    # 6. Open audio streams
    mic_stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=mic_rate,
        input=True,
        input_device_index=mic_index,
        frames_per_buffer=1024,
        stream_callback=mic_callback,
    )

    spk_stream = None
    if spk_index is not None:
        try:
            spk_stream = p.open(
                format=pyaudio.paFloat32,
                channels=2,
                rate=spk_rate,
                input=True,
                input_device_index=spk_index,
                frames_per_buffer=1024,
                stream_callback=speaker_callback,
            )
            log.info("Speaker loopback stream opened ✓")
        except Exception as exc:
            log.error(f"Speaker stream failed to open: {exc}")

    mic_stream.start_stream()
    if spk_stream:
        spk_stream.start_stream()

    # 7. Spawn inference workers (one per stream)
    stop_event     = threading.Event()
    mic_worker     = _make_worker(_mic_snap,     stop_event, "YOU",    "worker-YOU")
    speaker_worker = _make_worker(_speaker_snap, stop_event, "CLIENT", "worker-CLIENT")

    log.info(f"Recording — snapshot every {STEP_SECONDS}s (window={WINDOW_SECONDS}s) …")

    # 8. Main capture loop
    try:
        while True:
            time.sleep(STEP_SECONDS)

            mic_chunk = _drain(mic_buffer_ts)
            spk_chunk = _drain(speaker_buffer_ts)

            mic_window.extend(mic_chunk)
            if spk_chunk:
                speaker_window.extend(spk_chunk)

            _put_snapshot(_mic_snap, list(mic_window), mic_rate_global, "YOU")
            if spk_chunk:
                _put_snapshot(
                    _speaker_snap, list(speaker_window), speaker_rate_global, "CLIENT"
                )

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — stopping …")
        stop_event.set()

        mic_worker.join(timeout=10)
        speaker_worker.join(timeout=10)

        if mic_worker.is_alive():
            log.warning("Mic worker did not exit cleanly — daemon will auto-terminate")
        if speaker_worker.is_alive():
            log.warning("Speaker worker did not exit cleanly — daemon will auto-terminate")

    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        if spk_stream:
            spk_stream.stop_stream()
            spk_stream.close()
        p.terminate()

        log.info("Saving session audio …")
        save_full_audio()

        log.info(
            f"Skipped inference cycles — YOU: {_drop_counts.get('YOU', 0)}, "
            f"CLIENT: {_drop_counts.get('CLIENT', 0)}"
        )
        log.info(f"Transcript  : {TRANSCRIPT_FILE}")
        log.info(f"Mic audio   : {MIC_AUDIO_FILE}")
        log.info(f"Spk audio   : {SPEAKER_AUDIO_FILE}")


if __name__ == "__main__":
    main()