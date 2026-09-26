"""
Sample verse-level <audio, transcript> pairs for the forced-alignment survey.

For each of the 11 Open.Bible languages that were segmented with ReadAlongs
forced alignment (no Biblica timing files), this picks 50 verses from the
`train` split of davidguzmanr/open-bible-resources -- the data the TTS models
were trained on -- and writes:

  - audios/{language}/{BOOK}_{CCC}_Verse_{VVV}.wav   original 22.05 kHz clips
  - data/{language}.csv                              transcript + metadata

Selection ("typical speaking rate"): every released verse already passed the
outlier filter in open-bible-resources (utils/data_checks.py), which z-scores
the seconds-per-character ratio `duration / len(text)` per language and drops
anything beyond 3 standard deviations. We rank verses by |z| of that same
ratio, keep the TYPICAL_FRACTION closest to the language mean, and draw the
sample at random from that pool. This skips borderline segments while keeping
the sample varied across books, speakers and lengths.

Clean edges: the aligner cuts each verse exactly at its predicted boundary, with
no padding, so small boundary errors leave a syllable of the neighbouring verse
(or a clipped word) right at the start or end of the clip. The survey targets
word-level errors, so candidates are drawn from the pool in random order and
kept only if both the first and the last MIN_EDGE_SILENCE_MS of the clip are
silent (see edge_silence_ms). Rejected candidates are skipped until n verses
pass.

Results therefore describe the typical, cleanly cut part of the corpus, not a
uniform sample of it.

Verses used as reference recordings in the TTS listening test
(open-bible-surveys/audios/open-bible/{language}.csv) are excluded.

Each sampled verse is also tagged from the local USX text so annotations can be
broken down by the situations where forced alignment is most likely to fail:
  - is_first_verse   first verse of a chapter (follows the chapter-intro
                     placeholder that absorbs the spoken chapter announcement)
  - heading_before   a section heading / Psalm title sits right before the verse
  - heading_after    a section heading sits right after the verse
  - is_verse_range   the USX verse is a merged range (e.g. 3-4)
Headings are not part of the aligned text, so if the narrator reads them they
can leak into a neighbouring verse as extra words.

Usage:
    python scripts/sample_verses.py --all
    python scripts/sample_verses.py --language Swahili --n 50
"""

import argparse
import io
import re
import wave
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Audio, load_dataset

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

HF_REPO = "davidguzmanr/open-bible-resources"
SPLIT = "train"

# The 11 languages aligned with ReadAlongs (open-bible-resources README).
LANGUAGES = [
    "Arabic Standard",
    "Chichewa",
    "Dawro",
    "Gamo",
    "Gofa",
    "Haitian Creole",
    "Kikuyu",
    "Luganda",
    "Shona",
    "Swahili",
    "Turkish",
]

N_SAMPLES = 50
TYPICAL_FRACTION = 0.20
SEED = 42

# Edge-silence filter. Frame energy is measured with FRAME_MS windows every
# HOP_MS. A frame is "sound" if it is louder than the higher of
#   noise floor (10th percentile of frame dB) + NOISE_MARGIN_DB   and
#   speech level (95th percentile) - SPEECH_RANGE_DB,
# so the threshold adapts to recordings with audible room noise (e.g. Turkish,
# whose pauses sit around -60 dBFS) as well as to digitally silent ones.
MIN_EDGE_SILENCE_MS = 100
FRAME_MS = 20
HOP_MS = 10
NOISE_MARGIN_DB = 15.0
SPEECH_RANGE_DB = 40.0

REPO_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_ROOT = REPO_ROOT.parent / "open-bible-resources"
SURVEYS_ROOT = REPO_ROOT.parent / "open-bible-surveys"

# Paragraph styles that are spoken-but-not-aligned candidates: section headings,
# major section headings, Psalm titles and speaker labels. Cross-references
# (r, sr, mr) are not normally read aloud, so they are not counted.
HEADING_STYLES = {"s", "s1", "s2", "s3", "s4", "ms", "ms1", "ms2", "ms3", "d", "sp", "qa"}

VERSE_SID_RE = re.compile(r"^\s*([A-Z0-9]+)\s+(\d+):(\d+)(?:([-,])\d+)?")
CHAPTER_SID_RE = re.compile(r"^\s*([A-Z0-9]+)\s+(\d+)\s*$")


# -----------------------------------------------------------------------------
# USX tags
# -----------------------------------------------------------------------------

def find_usx_dir(language: str) -> Path | None:
    """USX directory with the most books (Hausa-style USX_1/USX_2 splits)."""
    release = RESOURCES_ROOT / "data" / "texts" / language / "USX" / "release"
    candidates = [d for d in release.glob("USX_*") if d.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: len(list(d.glob("*.usx"))))


def usx_verse_tags(usx_file: Path) -> dict[tuple[int, int], dict]:
    """
    Map (chapter, verse) -> {heading_before, heading_after, is_verse_range}.

    Walks the USX in document order. A heading paragraph marks the next verse of
    the same chapter as `heading_before` and the previous one as `heading_after`.
    Merged verses (sid "GEN 1:3-4") are keyed by their first verse number, which
    is how force_align_book.py names the output file.
    """
    root = ET.parse(usx_file).getroot()
    tags: dict[tuple[int, int], dict] = {}
    chapter = None
    last_verse = None       # (chapter, verse) of the most recent verse start
    pending_heading = False  # heading seen since last verse start

    for elem in root.iter():
        if elem.tag == "chapter" and "sid" in elem.attrib:
            m = CHAPTER_SID_RE.match(elem.attrib["sid"])
            if m:
                chapter = int(m.group(2))
                last_verse = None
                pending_heading = False
        elif elem.tag == "para" and elem.attrib.get("style") in HEADING_STYLES:
            if chapter is not None and "".join(elem.itertext()).strip():
                pending_heading = True
                if last_verse is not None:
                    tags[last_verse]["heading_after"] = True
        elif elem.tag == "verse" and "sid" in elem.attrib:
            m = VERSE_SID_RE.match(elem.attrib["sid"])
            if not m:
                continue
            key = (int(m.group(2)), int(m.group(3)))
            tags[key] = {
                "heading_before": pending_heading,
                "heading_after": False,
                "is_verse_range": m.group(4) is not None,
            }
            last_verse = key
            pending_heading = False
    return tags


def add_usx_tags(sample: pd.DataFrame, language: str) -> pd.DataFrame:
    sample = sample.copy()
    sample["is_first_verse"] = sample["verse"].astype(int) == 1
    for col in ("heading_before", "heading_after", "is_verse_range"):
        sample[col] = pd.NA

    usx_dir = find_usx_dir(language)
    if usx_dir is None:
        print(f"  WARNING: no local USX for {language}; heading/range tags left empty.", file=sys.stderr)
        return sample

    cache: dict[str, dict] = {}
    for idx, row in sample.iterrows():
        code = row["filename"].split("_")[0]
        if code not in cache:
            usx_file = usx_dir / f"{code}.usx"
            cache[code] = usx_verse_tags(usx_file) if usx_file.exists() else {}
        t = cache[code].get((int(row["chapter"]), int(row["verse"])))
        if t is not None:
            for col, value in t.items():
                sample.at[idx, col] = value
    return sample


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------

def tts_reference_verses(language: str) -> set[str]:
    """Filenames already used as ground truth in the TTS listening test."""
    csv_path = SURVEYS_ROOT / "audios" / "open-bible" / f"{language}.csv"
    if not csv_path.exists():
        return set()
    return set(pd.read_csv(csv_path)["filename"])


def edge_silence_ms(wav_bytes: bytes) -> tuple[int, int]:
    """Milliseconds of silence before the first and after the last sound frame."""
    with wave.open(io.BytesIO(wav_bytes)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, "expected 16-bit mono WAV"
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    win, hop = sr * FRAME_MS // 1000, sr * HOP_MS // 1000
    if len(x) < win:
        return 0, 0
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    threshold = max(np.percentile(db, 10) + NOISE_MARGIN_DB, np.percentile(db, 95) - SPEECH_RANGE_DB)
    sound = np.flatnonzero(db > threshold)
    if len(sound) == 0:
        return 0, 0
    return int(sound[0] * HOP_MS), int((len(db) - 1 - sound[-1]) * HOP_MS)


def typical_pool(df: pd.DataFrame, n: int, fraction: float) -> pd.DataFrame:
    """The `fraction` of rows whose speaking rate is closest to the language mean."""
    df = df.copy()
    df["lens_ratio"] = df["duration_seconds"] / df["text"].str.len()
    df["lens_ratio_z"] = (df["lens_ratio"] - df["lens_ratio"].mean()) / df["lens_ratio"].std()
    pool = df.loc[df["lens_ratio_z"].abs().rank(method="first") <= max(n, round(fraction * len(df)))]
    print(f"  Typical pool: {len(pool)} / {len(df)} verses (|z| <= {pool['lens_ratio_z'].abs().max():.3f})")
    return pool


def select_clean_edges(ds, pool: pd.DataFrame, n: int, seed: int, min_silence_ms: int) -> tuple[pd.DataFrame, dict]:
    """
    Walk the pool in a seeded random order and keep the first n verses whose
    clip starts and ends with at least `min_silence_ms` of silence. Returns the
    sample (with lead/trail silence columns) and {filename: wav bytes}.
    """
    kept, audio_bytes, checked = [], {}, 0
    for _, row in pool.sample(frac=1, random_state=seed).iterrows():
        audio = ds[int(row["row_idx"])]["audio"]
        assert Path(audio["path"]).name == row["filename"], (audio["path"], row["filename"])
        checked += 1
        lead, trail = edge_silence_ms(audio["bytes"])
        if lead >= min_silence_ms and trail >= min_silence_ms:
            kept.append({**row.to_dict(), "lead_silence_ms": lead, "trail_silence_ms": trail})
            audio_bytes[row["filename"]] = audio["bytes"]
            if len(kept) == n:
                break
    print(f"  Clean edges (>= {min_silence_ms} ms silence at both ends): kept {len(kept)} of {checked} checked "
          f"({100 * len(kept) / checked:.0f}%)")
    if len(kept) < n:
        print(f"  WARNING: only {len(kept)} verses in the pool pass the edge-silence filter.", file=sys.stderr)
    return pd.DataFrame(kept).sort_values("filename"), audio_bytes


def process_language(language: str, args: argparse.Namespace) -> None:
    # decode=False keeps the original WAV bytes (no resampling or re-encoding).
    ds = load_dataset(HF_REPO, language, split=SPLIT).cast_column("audio", Audio(decode=False))

    # Filenames live in the audio struct; read them from the memory-mapped Arrow
    # chunks so no audio bytes are loaded.
    filenames = [p for chunk in ds.data.column("audio").chunks for p in chunk.field("path").to_pylist()]
    df = ds.select_columns(
        ["text", "testament", "book", "chapter", "verse", "duration_seconds", "speaker_id"]
    ).to_pandas()
    df["filename"] = filenames
    df["row_idx"] = range(len(df))
    print(f"  {SPLIT}: {len(df)} verses")

    excluded = tts_reference_verses(language)
    overlap = df["filename"].isin(excluded)
    if overlap.any():
        print(f"  Excluding {overlap.sum()} verses used in the TTS listening test")
    df = df[~overlap]

    pool = typical_pool(df, n=args.n, fraction=args.typical_fraction)
    sample, audio_bytes = select_clean_edges(
        ds, pool, n=args.n, seed=args.seed, min_silence_ms=args.min_edge_silence_ms
    )
    sample = add_usx_tags(sample, language)

    audio_dir = REPO_ROOT / "audios" / language
    audio_dir.mkdir(parents=True, exist_ok=True)
    for f in audio_dir.glob("*.wav"):
        f.unlink()
    for filename in sample["filename"]:
        (audio_dir / filename).write_bytes(audio_bytes[filename])
    print(f"  Saved {len(sample)} clips to {audio_dir.relative_to(REPO_ROOT)}/")

    columns = [
        "filename", "text", "testament", "book", "chapter", "verse", "duration_seconds",
        "speaker_id", "row_idx", "lens_ratio", "lens_ratio_z", "lead_silence_ms", "trail_silence_ms",
        "is_first_verse", "heading_before", "heading_after", "is_verse_range",
    ]
    csv_path = REPO_ROOT / "data" / f"{language}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sample[columns].to_csv(csv_path, index=False, encoding="utf-8")
    tags = sample[["is_first_verse", "heading_before", "heading_after", "is_verse_range"]]
    print(f"  Wrote {csv_path.relative_to(REPO_ROOT)}  tags: {tags.eq(True).sum().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", choices=LANGUAGES, help="Language to sample (or use --all).")
    parser.add_argument("--all", action="store_true", help="Sample all 11 languages.")
    parser.add_argument("--n", type=int, default=N_SAMPLES, help=f"Verses per language (default: {N_SAMPLES}).")
    parser.add_argument(
        "--typical-fraction", type=float, default=TYPICAL_FRACTION,
        help=f"Share of verses closest to the mean speaking rate to sample from (default: {TYPICAL_FRACTION}).",
    )
    parser.add_argument(
        "--min-edge-silence-ms", type=int, default=MIN_EDGE_SILENCE_MS,
        help=f"Silence required at both ends of a clip; 0 disables the filter (default: {MIN_EDGE_SILENCE_MS}).",
    )
    parser.add_argument("--seed", type=int, default=SEED, help=f"Random seed (default: {SEED}).")
    args = parser.parse_args()

    if not args.language and not args.all:
        parser.error("Specify --language <LANG> or --all.")

    for language in LANGUAGES if args.all else [args.language]:
        print(f"\n{'=' * 60}\n{language}\n{'=' * 60}")
        process_language(language, args)


if __name__ == "__main__":
    main()
