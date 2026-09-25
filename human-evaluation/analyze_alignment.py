"""
Summarise the forced-alignment human evaluation (BibleTTS Sec. 4.3 protocol).

Reads the anonymized Label Studio exports (annotations-anonymized/{lang}.csv),
joins them to the tracking CSVs (humanalign/{lang}/tracking_{lang}.csv) and
reports, per language, the share of verses whose majority label is:

  EM        No missing or extra words (exact match)
  Add.      Audio contains EXTRA words not in the transcript
  Miss.     Audio is MISSING words that are in the transcript
  Both      Audio is MISSING words AND includes EXTRA words
  Conflict  no label has a strict plurality (e.g. three different labels)

as in Table 3 of the paper, plus Krippendorff's alpha (nominal) for
inter-annotator agreement. It also breaks EM down by the alignment-risk tags
from scripts/sample_verses.py and counts the optional "where" answers.

Outputs (in human-evaluation/results/):
  - alignment_summary.csv   one row per language (+ pooled)
  - alignment_by_verse.csv  one row per verse with its votes and majority label
  - alignment_by_tag.csv    EM / Add / Miss / Both by tag, pooled over languages

Usage:
    python human-evaluation/analyze_alignment.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = ROOT / "human-evaluation" / "annotations-anonymized"
TRACKING_DIR = ROOT / "humanalign"
RESULTS_DIR = ROOT / "human-evaluation" / "results"

LABELS = {
    "No missing or extra words (exact match)": "EM",
    "No missing or extra words": "EM",  # wording before "(exact match)" was added
    "Audio contains EXTRA words not in the transcript": "Add.",
    "Audio is MISSING words that are in the transcript": "Miss.",
    "Audio is MISSING words AND includes EXTRA words": "Both",
}
CATEGORIES = ["EM", "Add.", "Miss.", "Both", "Conflict"]
TAGS = ["is_first_verse", "heading_before", "heading_after", "is_verse_range"]


def parse_choices(value) -> list[str]:
    """Label Studio CSV exports multi-choice answers as JSON; single ones as plain text."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [value]
    if isinstance(parsed, dict):
        return list(parsed.get("choices", []))
    return parsed if isinstance(parsed, list) else [str(parsed)]


def majority(votes: list[str]) -> str:
    """Label with a strict plurality; otherwise 'Conflict' (paper: evenly spread votes)."""
    counts = Counter(votes).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return "Conflict"
    return counts[0][0]


def krippendorff_alpha_nominal(units: list[list[str]]) -> float:
    """Krippendorff's alpha for nominal data; units with < 2 ratings are ignored."""
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    values = sorted({v for u in units for v in u})
    index = {v: i for i, v in enumerate(values)}
    coincidence = np.zeros((len(values), len(values)))
    for u in units:
        counts = np.bincount([index[v] for v in u], minlength=len(values)).astype(float)
        coincidence += (np.outer(counts, counts) - np.diag(counts)) / (len(u) - 1)
    n_c = coincidence.sum(axis=1)
    n = n_c.sum()
    observed = n - np.trace(coincidence)
    expected = (n * n - (n_c ** 2).sum()) / (n - 1)
    return 1.0 - observed / expected if expected > 0 else 1.0


def load_language(csv_path: Path) -> pd.DataFrame:
    language = csv_path.stem
    ann = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    ann = ann[ann["alignment"].isin(LABELS)].copy()
    ann["label"] = ann["alignment"].map(LABELS)
    ann["location"] = ann.get("location", pd.Series("", index=ann.index)).apply(parse_choices)
    ann["language"] = language

    tracking_path = TRACKING_DIR / language / f"tracking_{language.replace(' ', '_')}.csv"
    tracking = pd.read_csv(tracking_path, dtype={"task_uid": str, "chapter": str, "verse": str})
    return ann.merge(tracking, on="task_uid", how="left", validate="many_to_one")


def summarise(verses: pd.DataFrame, votes: list[list[str]]) -> dict:
    shares = verses["majority"].value_counts(normalize=True).reindex(CATEGORIES, fill_value=0) * 100
    return {
        "verses": len(verses),
        "annotations": sum(len(v) for v in votes),
        **{c: round(shares[c], 1) for c in CATEGORIES},
        "alpha": round(krippendorff_alpha_nominal(votes), 3),
    }


def main() -> None:
    csv_files = sorted(ANNOTATIONS_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No anonymized annotations in {ANNOTATIONS_DIR}")

    ann = pd.concat([load_language(p) for p in csv_files], ignore_index=True)
    keys = ["language", "task_uid"]
    verses = (
        ann.groupby(keys)
        .agg(
            filename=("filename", "first"),
            votes=("label", list),
            locations=("location", lambda s: sorted({x for xs in s for x in xs})),
            **{t: (t, "first") for t in TAGS},
        )
        .reset_index()
    )
    verses["n_votes"] = verses["votes"].str.len()
    verses["majority"] = verses["votes"].apply(majority)

    rows = []
    for language, group in verses.groupby("language"):
        rows.append({"language": language, **summarise(group, group["votes"].tolist())})
    rows.append({"language": "ALL", **summarise(verses, verses["votes"].tolist())})
    summary = pd.DataFrame(rows)

    tag_rows = []
    for tag in TAGS:
        flags = verses[tag].astype(str).str.lower().eq("true")
        for value, group in ((True, verses[flags]), (False, verses[~flags])):
            if len(group):
                shares = group["majority"].value_counts(normalize=True).reindex(CATEGORIES, fill_value=0) * 100
                tag_rows.append({"tag": tag, "value": value, "verses": len(group),
                                 **{c: round(shares[c], 1) for c in CATEGORIES}})
    by_tag = pd.DataFrame(tag_rows)

    location_counts = Counter(x for xs in ann["location"] for x in xs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "alignment_summary.csv", index=False)
    verses.to_csv(RESULTS_DIR / "alignment_by_verse.csv", index=False)
    by_tag.to_csv(RESULTS_DIR / "alignment_by_tag.csv", index=False)

    print("Majority label per verse (% of verses); alpha = Krippendorff's alpha (nominal)\n")
    print(summary.to_markdown(index=False))
    print("\nBy alignment-risk tag (pooled over languages)\n")
    print(by_tag.to_markdown(index=False))
    print("\nWhere the problem is (optional answers, all annotations):")
    for loc, n in location_counts.most_common():
        print(f"  {loc}: {n}")
    uneven = verses[verses["n_votes"] != verses["n_votes"].mode().iat[0]]
    if len(uneven):
        print(f"\nNote: {len(uneven)} verse(s) have an unusual number of ratings "
              f"({sorted(uneven['n_votes'].unique())}).")
    print(f"\nResults written to {RESULTS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
