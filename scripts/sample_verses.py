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
the sample varied across books, speakers and lengths. Results therefore
describe the typical part of the corpus, not a uniform sample of it.

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
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

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


def select_typical(df: pd.DataFrame, n: int, fraction: float, seed: int) -> pd.DataFrame:
    """Random n from the `fraction` of rows whose speaking rate is closest to the mean."""
    df = df.copy()
    df["lens_ratio"] = df["duration_seconds"] / df["text"].str.len()
    df["lens_ratio_z"] = (df["lens_ratio"] - df["lens_ratio"].mean()) / df["lens_ratio"].std()
    pool = df.loc[df["lens_ratio_z"].abs().rank(method="first") <= max(n, round(fraction * len(df)))]
    print(f"  Typical pool: {len(pool)} / {len(df)} verses (|z| <= {pool['lens_ratio_z'].abs().max():.3f})")
    return pool.sample(n=n, random_state=seed).sort_values("filename")


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

    sample = select_typical(df, n=args.n, fraction=args.typical_fraction, seed=args.seed)
    sample = add_usx_tags(sample, language)

    audio_dir = REPO_ROOT / "audios" / language
    audio_dir.mkdir(parents=True, exist_ok=True)
    for f in audio_dir.glob("*.wav"):
        f.unlink()
    for row_idx, filename in zip(sample["row_idx"], sample["filename"]):
        audio = ds[int(row_idx)]["audio"]
        assert Path(audio["path"]).name == filename, (audio["path"], filename)
        (audio_dir / filename).write_bytes(audio["bytes"])
    print(f"  Saved {len(sample)} clips to {audio_dir.relative_to(REPO_ROOT)}/")

    columns = [
        "filename", "text", "testament", "book", "chapter", "verse", "duration_seconds",
        "speaker_id", "row_idx", "lens_ratio", "lens_ratio_z",
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
    parser.add_argument("--seed", type=int, default=SEED, help=f"Random seed (default: {SEED}).")
    args = parser.parse_args()

    if not args.language and not args.all:
        parser.error("Specify --language <LANG> or --all.")

    for language in LANGUAGES if args.all else [args.language]:
        print(f"\n{'=' * 60}\n{language}\n{'=' * 60}")
        process_language(language, args)


if __name__ == "__main__":
    main()
