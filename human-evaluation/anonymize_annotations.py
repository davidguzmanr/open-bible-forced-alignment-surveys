"""
Anonymize and deduplicate all annotation CSV files.

Reads CSVs from human-evaluation/annotations/, replaces each unique annotator
email with a stable anonymous ID (annotator_1, annotator_2, ...) assigned in
order of first appearance across all files, drops duplicated annotations
(same annotator rating the same audio more than once, keeping the most recent
one according to 'updated_at'), and writes the results to
human-evaluation/annotations-anonymized/.
"""

from pathlib import Path

import pandas as pd

INPUT_DIR = Path(__file__).parent / "annotations"
OUTPUT_DIR = Path(__file__).parent / "annotations-anonymized"
ANNOTATOR_COL = "annotator"
AUDIO_COL = "audio_url"
UPDATED_AT_COL = "updated_at"


def build_annotator_map(dataframes: dict[Path, pd.DataFrame]) -> dict[str, str]:
    """Scan all files and assign a stable anonymous ID to each unique annotator."""
    seen: dict[str, str] = {}
    for df in dataframes.values():
        for email in df[ANNOTATOR_COL]:
            if email not in seen:
                seen[email] = f"annotator_{len(seen) + 1}"
    return seen


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent annotation per (annotator, audio_url) pair."""
    # Sort on a temporary column so 'updated_at' keeps its original string format.
    return (
        df.assign(_updated_at=pd.to_datetime(df[UPDATED_AT_COL]))
        .sort_values("_updated_at", ascending=False)
        .drop_duplicates(subset=[ANNOTATOR_COL, AUDIO_COL], keep="first")
        .sort_index()
        .reset_index(drop=True)
        .drop(columns="_updated_at")
    )


def main() -> None:
    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

    print(f"Found {len(csv_files)} file(s) in {INPUT_DIR}")

    # Read everything as strings so values are written back verbatim (e.g. the
    # leading zeros of 'task_uid' are not stripped by a numeric conversion).
    dataframes = {
        path: pd.read_csv(path, dtype=str, keep_default_na=False)
        for path in csv_files
    }

    mapping = build_annotator_map(dataframes)
    print(f"\nAnnotator mapping ({len(mapping)} unique annotators):")
    for email, anon_id in mapping.items():
        print(f"  {email!r:45s} -> {anon_id}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for src, df in dataframes.items():
        df[ANNOTATOR_COL] = df[ANNOTATOR_COL].map(mapping)

        n_before = len(df)
        df = deduplicate(df)
        print(f"\n{src.name}: {n_before} rows -> {len(df)} rows after deduplication")
        print(df[ANNOTATOR_COL].value_counts().to_string())

        dst = OUTPUT_DIR / src.name
        df.to_csv(dst, index=False)
        print(f"Saved: {dst}")

    print("\nDone.")


if __name__ == "__main__":
    main()
