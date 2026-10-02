"""
Build an English practice tutorial for the HumanSignal alignment task.

Annotators label a handful of English clips with the same interface as the real
survey, then open "Show the correct answer" to check themselves. The clips come
from LJSpeech (davidguzmanr/CSS10-Multilingual-LJSpeech, "English"), whose
consecutive utterances are consecutive sentences of the same book, so the
"neighbouring verse" errors are made from the real previous / next sentence:

  1  exact match            one utterance as it is
  2  EXTRA (end)            utterance + the first words of the next one
  3  MISSING (end)          utterance with its last words cut off
  4  MISSING + EXTRA        last word of the previous utterance + utterance
                            with its last words cut off
  5  exact match            the last syllable of the previous utterance at the
                            start ("-ry" of "necessary")
  6  exact match            the end of the last word slightly clipped
  7  exact match            both of the above at once

Word cut points come from ReadAlongs word alignments (English g2p, audio
resampled to 16 kHz for alignment only). Edge fragments and clipping use the
same energy-based speech start/end as scripts/sample_verses.py.

Writes:
  - tutorial/audios/practice_{NN}.wav                  (neutral names: no answer hints)
  - tutorial/humanalign/answers_tutorial.csv          (filename -> example and answer)
  - tutorial/humanalign/labeling_config_tutorial.xml   (real config + answer panel)
  - tutorial/humanalign/tasks_tutorial.json            (fixed teaching order)

Usage:
    python tutorial/build_tutorial.py
    python tutorial/build_tutorial.py --readalongs /path/to/envs/ReadAlongs/bin/readalongs
"""

import argparse
import csv
import html
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from urllib.parse import quote

import numpy as np
from datasets import Audio, load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "surveys"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_survey_alignment import (  # noqa: E402
    ALIGNMENT_CHOICES,
    build_instructions_html,
    build_labeling_config,
    build_transcript_html,
)
from sample_verses import edge_silence_ms  # noqa: E402

OUT = ROOT / "tutorial"
DATASET = "davidguzmanr/CSS10-Multilingual-LJSpeech"
# Every utterance used is in the first train shard; loading only it avoids
# downloading the whole English config.
SHARD = "English/train-00000-of-00008.parquet"
SR = 22050

AUDIO_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/davidguzmanr/open-bible-forced-alignment-surveys/"
    "refs/heads/{ref}/tutorial/audios/{filename}"
)

EXACT, EXTRA, MISSING, BOTH = ALIGNMENT_CHOICES

PAUSE_S = 0.30        # pause inserted between joined utterances
FRAGMENT_S = 0.10     # piece of the previous utterance left at the start (example 7)
SYLLABLE_S = 0.18     # final syllable of the previous utterance (example 5)
CLIP_S = 0.07         # how much of the last word is shaved off
FADE_S = 0.005        # short fades at hard cuts to avoid clicks


# -----------------------------------------------------------------------------
# Audio helpers
# -----------------------------------------------------------------------------

def to_array(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)


def to_wav(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.clip(x, -32768, 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def seg(x: np.ndarray, start: float | None = None, end: float | None = None) -> np.ndarray:
    """x[start:end] in seconds, with short fades at both cuts."""
    y = x[None if start is None else int(start * SR): None if end is None else int(end * SR)].copy()
    n = min(int(FADE_S * SR), len(y) // 2)
    if n:
        y[:n] *= np.linspace(0, 1, n)
        y[-n:] *= np.linspace(1, 0, n)
    return y


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def speech_bounds(x: np.ndarray) -> tuple[float, float]:
    """Start and end of speech (s), using the sampler's edge-silence measure."""
    lead, trail = edge_silence_ms(to_wav(x))
    return lead / 1000, len(x) / SR - trail / 1000


# -----------------------------------------------------------------------------
# Word alignment (ReadAlongs)
# -----------------------------------------------------------------------------

WORD_RE = re.compile(r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"')


def word_times(readalongs: str, x: np.ndarray, text: str) -> list[tuple[float, float, str]]:
    """(start, end, word) for every word, from a ReadAlongs TextGrid."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.wav").write_bytes(to_wav(x))
        # Align at 16 kHz: SoundSwallower's timestamps drift at 22.05 kHz.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp / "in.wav", "-ar", "16000",
                        tmp / "in16.wav"], check=True)
        (tmp / "text.txt").write_text(text + "\n", encoding="utf-8")
        subprocess.run([readalongs, "align", "-l", "eng", "-o", "textgrid", "-f",
                        tmp / "text.txt", tmp / "in16.wav", tmp / "out"], check=True, capture_output=True)
        tg = next((tmp / "out").glob("*.TextGrid")).read_text(encoding="utf-8")
    words_tier = tg[tg.index('name = "Word"'):]
    return [(float(a), float(b), w) for a, b, w in WORD_RE.findall(words_tier) if w.strip()]


def cut_before(x: np.ndarray, words: list, k: int, window: float = 0.06) -> float:
    """
    Cut point between word k-1 and word k: the quietest 10 ms frame within
    `window` s of the aligner's boundary. ReadAlongs returns back-to-back word
    intervals, so cutting exactly at the boundary can keep the onset of word k.
    """
    boundary = (words[k - 1][1] + words[k][0]) / 2
    hop = int(0.01 * SR)
    lo, hi = int((boundary - window) * SR), int((boundary + window) * SR)
    starts = range(max(lo, 0), min(hi, len(x) - hop), hop)
    energy = [float(np.mean(x[i:i + hop] ** 2)) for i in starts]
    return (starts[int(np.argmin(energy))] + hop / 2) / SR


# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------

def answer_html(choice: str, where: str, explanation: str) -> str:
    where_html = f"<br><strong>Where:</strong> {where}" if where else ""
    return (
        f'<div style="padding:4px 0;"><strong>Correct answer:</strong> {html.escape(choice)}'
        f"{where_html}<p style=\"margin:6px 0 0;\">{explanation}</p></div>"
    )


def build_examples(utts: dict, readalongs: str) -> list[dict]:
    a = {k: to_array(v["bytes"]) for k, v in utts.items()}
    t = {k: v["text"] for k, v in utts.items()}
    examples = []

    # 1. Exact match.
    examples.append({
        "slug": "exact_match", "audio": a["LJ001-0043"], "text": t["LJ001-0043"],
        "answer": answer_html(EXACT, "", "Every word of the transcript is spoken, and nothing else."),
    })

    # 2. EXTRA at the end: the next sentence starts "Even in Italy ...".
    nxt = word_times(readalongs, a["LJ001-0062"], t["LJ001-0062"])
    assert [x[2].lower() for x in nxt[:3]] == ["even", "in", "italy"], nxt[:3]
    extra_end = cut_before(a["LJ001-0062"], nxt, 3)  # after "Italy"
    examples.append({
        "slug": "extra_end",
        "audio": np.concatenate([a["LJ001-0061"], silence(PAUSE_S), seg(a["LJ001-0062"], None, extra_end)]),
        "text": t["LJ001-0061"],
        "answer": answer_html(
            EXTRA, "At the end of the audio",
            "After the transcript ends, the speaker continues with <em>“Even in Italy”</em>, "
            "the start of the next sentence. Those are whole extra words."),
    })

    # 3. MISSING at the end: "... fine printing in Italy." loses "in Italy".
    w = word_times(readalongs, a["LJ001-0053"], t["LJ001-0053"])
    k = len(w) - 2
    assert [x[2].lower() for x in w[k:]] == ["in", "italy"], w[k:]
    examples.append({
        "slug": "missing_end",
        "audio": seg(a["LJ001-0053"], None, cut_before(a["LJ001-0053"], w, k)),
        "text": t["LJ001-0053"],
        "answer": answer_html(
            MISSING, "At the end of the audio",
            "The audio stops after <em>“printing”</em>: the last words of the transcript, "
            "<em>“in Italy”</em>, are never spoken."),
    })

    # 4. EXTRA at the start + MISSING at the end.
    prev = word_times(readalongs, a["LJ001-0010"], t["LJ001-0010"])
    assert prev[-1][2].lower() == "letterpress", prev[-1]
    w = word_times(readalongs, a["LJ001-0011"], t["LJ001-0011"])
    k = len(w) - 2
    assert [x[2].lower() for x in w[k:]] == ["in", "form"], w[k:]
    examples.append({
        "slug": "missing_and_extra",
        "audio": np.concatenate([
            seg(a["LJ001-0010"], cut_before(a["LJ001-0010"], prev, len(prev) - 1)),
            silence(PAUSE_S),
            seg(a["LJ001-0011"], None, cut_before(a["LJ001-0011"], w, k)),
        ]),
        "text": t["LJ001-0011"],
        "answer": answer_html(
            BOTH, "At the start of the audio; at the end of the audio",
            "The clip starts with <em>“letterpress”</em>, the last word of the previous "
            "sentence (extra), and stops before <em>“in form”</em> (missing)."),
    })

    # 5. Exact match despite a stray syllable at the start: the previous sentence
    #    ends "... renders necessary." and the clip keeps its final "-ry". Taking
    #    the whole "-sary" would sound like the word "sorry", so keep it shorter.
    p_start, p_end = speech_bounds(a["LJ001-0129"])
    examples.append({
        "slug": "exact_syllable_at_start",
        "audio": np.concatenate([
            seg(a["LJ001-0129"], p_end - SYLLABLE_S, p_end), silence(PAUSE_S), a["LJ001-0130"],
        ]),
        "text": t["LJ001-0130"],
        "answer": answer_html(
            EXACT, "",
            "The clip starts with a short syllable, the end of <em>\u201cnecessary\u201d</em>, the "
            "last word of the previous sentence. It is only part of a word, not a whole word you can "
            "recognise, and every word of the transcript is spoken, so this is still an exact match."),
    })

    # 6. Exact match despite a slightly clipped last word.
    s_start, s_end = speech_bounds(a["LJ001-0123"])
    examples.append({
        "slug": "exact_clipped_end",
        "audio": seg(a["LJ001-0123"], None, s_end - CLIP_S),
        "text": t["LJ001-0123"],
        "answer": answer_html(
            EXACT, "",
            "The very end of <em>“hurry”</em> is cut a little short, but you can still "
            "recognise the word. Nothing is missing, so this is an exact match."),
    })

    # 7. Exact match with both edge artefacts at once.
    p_start, p_end = speech_bounds(a["LJ001-0024"])
    s_start, s_end = speech_bounds(a["LJ001-0025"])
    examples.append({
        "slug": "exact_sound_at_start_and_clipped_end",
        "audio": np.concatenate([
            seg(a["LJ001-0024"], p_end - FRAGMENT_S, p_end), silence(PAUSE_S),
            seg(a["LJ001-0025"], None, s_end - CLIP_S),
        ]),
        "text": t["LJ001-0025"],
        "answer": answer_html(
            EXACT, "",
            "There is a tiny sound from the previous sentence at the start and the last word "
            "(<em>“read”</em>) is slightly clipped, but every word of the transcript is "
            "spoken and recognisable and no other whole word is heard. This is an exact match."),
    })
    return examples


TUTORIAL_INTRO = (
    '<div style="background:#eef4ff;border-left:4px solid #3b6fd8;padding:8px 14px;margin:0 0 10px;'
    'border-radius:4px;color:#1d2a44;">'
    "<strong>Practice round (English).</strong> These examples show every possible answer. "
    "Label each one as you would in the real task, then open <em>Show the correct answer</em> "
    "at the bottom to check. In the real task the clips and transcripts are in your language."
    "</div>"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--readalongs", default=shutil.which("readalongs"),
                        help="Path to the readalongs CLI (default: the one on PATH).")
    parser.add_argument("--ref", default="dev/tutorial",
                        help="Git branch the audio URLs point to; the tutorial lives only on this branch "
                             "(default: dev/tutorial).")
    args = parser.parse_args()
    if not args.readalongs:
        parser.error("readalongs not found; pass --readalongs /path/to/readalongs")

    needed = {"LJ001-0010", "LJ001-0011", "LJ001-0024", "LJ001-0025", "LJ001-0043", "LJ001-0053",
              "LJ001-0061", "LJ001-0062", "LJ001-0123", "LJ001-0129", "LJ001-0130"}
    ds = load_dataset(DATASET, data_files={"train": SHARD}, split="train").cast_column("audio", Audio(decode=False))
    utts = {}
    for row in ds:
        name = Path(row["audio"]["path"]).stem
        if name in needed:
            utts[name] = {"bytes": row["audio"]["bytes"], "text": row["text"]}
    missing = needed - set(utts)
    assert not missing, f"not found in {SHARD}: {sorted(missing)}"

    examples = build_examples(utts, args.readalongs)

    audio_dir = OUT / "audios"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for f in audio_dir.glob("*.wav"):
        f.unlink()
    instructions = TUTORIAL_INTRO + build_instructions_html("English (practice)")
    tasks, answers = [], []
    for i, ex in enumerate(examples, start=1):
        filename = f"practice_{i:02d}.wav"
        (audio_dir / filename).write_bytes(to_wav(ex["audio"]))
        tasks.append({"data": {
            "task_uid": f"{i:05d}",
            "filename": filename,
            "transcript": ex["text"],
            "transcript_html": build_transcript_html(ex["text"]),
            "instructions_html": instructions,
            "answer_html": ex["answer"],
            "audio_url": AUDIO_URL_TEMPLATE.format(ref=quote(args.ref, safe="/"), filename=filename),
        }})
        answers.append({"filename": filename, "example": ex["slug"], "transcript": ex["text"]})
        print(f"  {filename:18s} {ex['slug']:38s} {len(ex['audio']) / SR:5.2f} s")

    hl = OUT / "humanalign"
    hl.mkdir(parents=True, exist_ok=True)
    (hl / "labeling_config_tutorial.xml").write_text(build_labeling_config(answer_panel=True), encoding="utf-8")
    (hl / "tasks_tutorial.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(hl / "answers_tutorial.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(answers[0]))
        writer.writeheader()
        writer.writerows(answers)
    print(f"Wrote {len(tasks)} tasks to {hl.relative_to(ROOT)}/ (audio URLs point to branch '{args.ref}')")


if __name__ == "__main__":
    main()
