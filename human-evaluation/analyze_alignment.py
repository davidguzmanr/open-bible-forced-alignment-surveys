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
inter-annotator agreement. That table uses only the `random` sample group, which
is a uniform sample of the corpus; the `clean_edges` group (clips with silence
at both ends) is reported separately, next to the random group. It also breaks
EM down by the alignment-risk tags from scripts/sample_verses.py, counts the
"where" answers, and reports per-annotator checks:
  - EM on clean edges   share of clean_edges clips the annotator labelled EM;
                        these clips should nearly all be exact matches
  - agreement           share of their labels that match the other annotators'
                        label, on tasks where those others all agree
Annotators below 70% on either are flagged for a closer look (not excluded).

Outputs (in human-evaluation/results/):
  - alignment_summary.csv   one row per language (+ pooled), random group only
  - alignment_by_group.csv  EM / mismatch / conflict per language and sample group
  - alignment_by_verse.csv  one row per verse with its votes and majority label
  - alignment_by_tag.csv    EM / Add / Miss / Both by tag, pooled, random group
  - annotators.csv          per-annotator checks

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
FLAG_BELOW = 70.0  # % EM on clean edges / % agreement below which an annotator is flagged


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
    # Exports also carry task-data fields such as `filename`; take them from the
    # tracking CSV so the merge does not produce filename_x / filename_y.
    ann = ann.drop(columns=[c for c in tracking.columns if c != "task_uid" and c in ann.columns])
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
    if "sample_group" not in ann:  # tracking files from before the clean_edges group
        ann["sample_group"] = "random"
    ann["sample_group"] = ann["sample_group"].fillna("random")
    keys = ["language", "task_uid"]
    verses = (
        ann.groupby(keys)
        .agg(
            filename=("filename", "first"),
            sample_group=("sample_group", "first"),
            votes=("label", list),
            locations=("location", lambda s: sorted({x for xs in s for x in xs})),
            **{t: (t, "first") for t in TAGS},
        )
        .reset_index()
    )
    verses["n_votes"] = verses["votes"].str.len()
    verses["majority"] = verses["votes"].apply(majority)

    random_verses = verses[verses["sample_group"] == "random"]
    rows = []
    for language, group in random_verses.groupby("language"):
        rows.append({"language": language, **summarise(group, group["votes"].tolist())})
    rows.append({"language": "ALL", **summarise(random_verses, random_verses["votes"].tolist())})
    summary = pd.DataFrame(rows)

    group_rows = []
    for (language, sample_group), group in [*verses.groupby(["language", "sample_group"]),
                                            *(( ("ALL", g), grp) for g, grp in verses.groupby("sample_group"))]:
        shares = group["majority"].value_counts(normalize=True).reindex(CATEGORIES, fill_value=0) * 100
        group_rows.append({"language": language, "sample_group": sample_group, "verses": len(group),
                           "EM": round(shares["EM"], 1),
                           "mismatch": round(shares[["Add.", "Miss.", "Both"]].sum(), 1),
                           "Conflict": round(shares["Conflict"], 1)})
    by_group = pd.DataFrame(group_rows)

    # Per-annotator checks.
    others_consensus = {}
    for key, group in ann.groupby(keys):
        for idx, row in group.iterrows():
            others = group.loc[group.index != idx, "label"].tolist()
            if len(others) >= 2 and len(set(others)) == 1:
                others_consensus[idx] = others[0]
    ann["others_agree_on"] = pd.Series(others_consensus)
    annot_rows = []
    for (language, annotator), group in ann.groupby(["language", "annotator"]):
        clean = group[group["sample_group"] == "clean_edges"]
        rand = group[group["sample_group"] == "random"]
        judged = group.dropna(subset=["others_agree_on"])
        row = {
            "language": language, "annotator": annotator, "tasks": len(group),
            "EM on clean edges": round(100 * clean["label"].eq("EM").mean(), 1) if len(clean) else float("nan"),
            "EM on random": round(100 * rand["label"].eq("EM").mean(), 1) if len(rand) else float("nan"),
            "agreement": round(100 * judged["label"].eq(judged["others_agree_on"]).mean(), 1) if len(judged) else float("nan"),
            "agreement tasks": len(judged),
        }
        reasons = [name for name in ("EM on clean edges", "agreement") if row[name] < FLAG_BELOW]
        row["flag"] = ", ".join(f"low {r}" for r in reasons)
        annot_rows.append(row)
    annotators = pd.DataFrame(annot_rows)

    tag_rows = []
    for tag in TAGS:
        flags = random_verses[tag].astype(str).str.lower().eq("true")
        for value, group in ((True, random_verses[flags]), (False, random_verses[~flags])):
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
    by_group.to_csv(RESULTS_DIR / "alignment_by_group.csv", index=False)
    annotators.to_csv(RESULTS_DIR / "annotators.csv", index=False)

    print("Majority label per verse, random group only (% of verses); alpha = Krippendorff's alpha (nominal)\n")
    print(summary.to_markdown(index=False))
    print("\nBy sample group (% of verses; mismatch = Add. + Miss. + Both)\n")
    print(by_group.to_markdown(index=False))
    print(f"\nAnnotators (flagged below {FLAG_BELOW:.0f}%)\n")
    print(annotators.to_markdown(index=False))
    print("\nBy alignment-risk tag (random group, pooled over languages)\n")
    print(by_tag.to_markdown(index=False))
    print("\nWhere the problem is (all annotations with a mismatch label):")
    for loc, n in location_counts.most_common():
        print(f"  {loc}: {n}")
    uneven = verses[verses["n_votes"] != verses["n_votes"].mode().iat[0]]
    if len(uneven):
        print(f"\nNote: {len(uneven)} verse(s) have an unusual number of ratings "
              f"({sorted(uneven['n_votes'].unique())}).")
    print(f"\nResults written to {RESULTS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
