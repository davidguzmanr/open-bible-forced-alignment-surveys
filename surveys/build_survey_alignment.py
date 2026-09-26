"""
Generate Label Studio / HumanAlign tasks for evaluating forced-alignment quality
of verse-level <audio, transcript> pairs, following Section 4.3 of the
BibleTTS paper (Meyer et al., 2022).

Each task shows one aligned verse: the transcript and its audio clip.
Annotators pick the one option that best describes how well they match:

  1. No missing or extra words (exact match)
  2. Audio contains EXTRA words not in the transcript
  3. Audio is MISSING words that are in the transcript
  4. Audio is MISSING words AND includes EXTRA words

For options 2-4 two follow-ups appear: where the problem is (start / middle /
end of the clip; required) and a free-text box for the words involved
(optional). The "where" choices carry their own visibleWhen condition so the
requirement only applies when they are shown.

One set of output files is created per language:
  - labeling_config_{lang}.xml  — paste into the Label Studio project config
  - tasks_{lang}.json           — import into the Label Studio project
  - tracking_{lang}.csv         — maps task_uid -> filename and verse metadata/tags

Inputs come from scripts/sample_verses.py (data/{lang}.csv, audios/{lang}/).

Usage:
    python surveys/build_survey_alignment.py --language Swahili
    python surveys/build_survey_alignment.py --all

    # Create project + import tasks directly via API:
    python surveys/build_survey_alignment.py --language Swahili \\
        --api-key YOUR_TOKEN --base-url https://app.humansignal.com
"""

import argparse
import csv
import html
import json
import random
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from romanize_arabic import romanize as romanize_arabic

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_DIR = Path("data")
AUDIO_DIR = Path("audios")

AUDIO_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/davidguzmanr/open-bible-forced-alignment-surveys/"
    "refs/heads/main/audios/{language}/{filename}"
)

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

# Options as worded in BibleTTS Section 4.3, with "(exact match)" added to make
# that category explicit. The paper lists it last; here it is shown first.
EXACT_MATCH = "No missing or extra words (exact match)"
MISMATCH_CHOICES = [
    "Audio contains EXTRA words not in the transcript",
    "Audio is MISSING words that are in the transcript",
    "Audio is MISSING words AND includes EXTRA words",
]
ALIGNMENT_CHOICES = [EXACT_MATCH, *MISMATCH_CHOICES]
LOCATION_CHOICES = [
    "At the start of the audio",
    "In the middle of the audio",
    "At the end of the audio",
]

TRACKING_COLUMNS = [
    "filename", "book", "chapter", "verse", "duration_seconds", "speaker_id",
    "lens_ratio_z", "lead_silence_ms", "trail_silence_ms", "is_first_verse", "heading_before", "heading_after", "is_verse_range",
]

DEFAULT_BASE_URL = "https://app.humansignal.com"

# Latin transcriptions stored in the task data as `transcript_latin` for
# following the audio in non-Latin scripts. The labeling config never references
# this field, so annotators are not shown it.
ROMANIZERS = {"Arabic Standard": romanize_arabic}


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_items(language: str) -> pd.DataFrame:
    """Sampled verses for a language, restricted to those with a WAV on disk."""
    csv_path = DATA_DIR / f"{language}.csv"
    if not csv_path.exists():
        sys.exit(f"ERROR: {csv_path} not found. Run scripts/sample_verses.py --language '{language}' first.")

    df = pd.read_csv(csv_path, dtype={"chapter": str, "verse": str})
    on_disk = {p.name for p in (AUDIO_DIR / language).glob("*.wav")}
    missing = ~df["filename"].isin(on_disk)
    if missing.any():
        print(f"  WARNING: {missing.sum()} sampled verse(s) have no WAV — skipped.", file=sys.stderr)
    return df[~missing].reset_index(drop=True)


def write_tracking_csv(records: list[dict], output_path: Path) -> None:
    if not records:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"  Tracking CSV written to:    {output_path}")


# -----------------------------------------------------------------------------
# Label Studio labeling config (XML)
# -----------------------------------------------------------------------------

def build_instructions_html(language: str) -> str:
    """
    Instructions shown at the top of every task.

    Label Studio requires <HyperText> to reference a data field via value="$...",
    so this string is stored in each task's data as `instructions_html`.
    """
    return (
        f'<center><h3>Audio–transcript match &#8212; {html.escape(language)}</h3></center>'
        '<p>Each clip is one Bible verse that was cut automatically from a full chapter recording. '
        'Listen to the <strong>whole clip</strong> while reading the transcript, and choose the '
        'option that best describes whether the <strong>spoken words</strong> match the '
        '<strong>written words</strong>.</p>'
        '<ul style="margin:4px 0 8px 18px;">'
        '<li><strong>EXTRA</strong>: you hear words that are not in the transcript '
        '(for example part of the previous or next verse, or a chapter number or section '
        'title read aloud).</li>'
        '<li><strong>MISSING</strong>: words in the transcript are not heard, or the first or '
        'last word is cut off so it cannot be understood.</li>'
        '</ul>'
        '<p style="margin:4px 0;">Please ignore voice quality, accent, speed, breaths and '
        'background noise. Punctuation is not spoken, and numbers written as digits may be read '
        'out as words; neither counts as a mismatch.</p>'
        '<p style="margin:4px 0;">If you choose one of the last three options, please also say '
        '<em>where</em> the problem is (required) and, if you can, write the extra or missing '
        'words (optional).</p>'
    )


def build_transcript_html(text: str) -> str:
    # dir="auto" renders right-to-left scripts (Arabic) correctly.
    return (
        '<div dir="auto" style="font-size:1.2em;line-height:1.6;">'
        f'{html.escape(text)}'
        '</div>'
    )


def build_labeling_config() -> str:
    """
    Return a Label Studio XML config for a single-clip alignment judgement.

    Each task exposes:
      $instructions_html — task instructions (rendered as HTML)
      $transcript_html   — the aligned verse text (rendered as HTML)
      $audio_url         — URL of the aligned verse clip
    """
    alignment = "\n".join(f'    <Choice value="{c}" />' for c in ALIGNMENT_CHOICES)
    location = "\n".join(f'      <Choice value="{c}" />' for c in LOCATION_CHOICES)
    mismatch_values = ",".join(MISMATCH_CHOICES)
    return f"""\
<View>
  <HyperText name="instructions" value="$instructions_html" />

  <View style="background:#f5f5f5;border-left:4px solid #aaa;padding:10px 16px;margin:12px 0;border-radius:4px;">
    <Header value="Transcript:" />
    <HyperText name="transcript" value="$transcript_html" />
  </View>

  <Audio name="audio" value="$audio_url" />

  <Header value="Which option best describes the audio?" />
  <Choices name="alignment" toName="audio" choice="single" required="true">
{alignment}
  </Choices>

  <View visibleWhen="choice-selected" whenTagName="alignment" whenChoiceValue="{mismatch_values}">
    <Header value="Where is the problem? (select all that apply)" />
    <Choices name="location" toName="audio" choice="multiple" showInline="true"
             required="true" requiredMessage="Please select where the extra or missing words are."
             visibleWhen="choice-selected" whenTagName="alignment" whenChoiceValue="{mismatch_values}">
{location}
    </Choices>
    <Header value="Which words are extra or missing? (optional)" />
    <TextArea name="words" toName="audio" rows="2" editable="true" maxSubmissions="1"
              placeholder="e.g. extra: 'Chapter 5' at the start" />
  </View>
</View>"""


# -----------------------------------------------------------------------------
# Tasks
# -----------------------------------------------------------------------------

def build_tasks(
    items: pd.DataFrame,
    language: str,
    instructions_html: str,
    seed: int = 42,
    tracking_csv_path: Path | None = None,
) -> list[dict]:
    """
    Shuffle the sampled verses and return Label Studio task dicts.

    Each task data contains: task_uid, filename, transcript, transcript_html,
    instructions_html, audio_url (plus transcript_latin for ROMANIZERS languages,
    which the labeling config does not display). The tracking CSV keeps the verse metadata and
    alignment-risk tags (joined to exported annotations via task_uid).
    """
    rows = items.to_dict("records")
    random.Random(seed).shuffle(rows)
    romanize = ROMANIZERS.get(language)
    print(f"  Total tasks: {len(rows)} (shuffled with seed={seed})")

    tasks: list[dict] = []
    tracking_records: list[dict] = []
    for idx, row in enumerate(rows):
        task_uid = f"{idx:05d}"
        data = {
            "task_uid":          task_uid,
            "filename":          row["filename"],
            "transcript":        row["text"],
        }
        if romanize:
            data["transcript_latin"] = romanize(row["text"])
        data.update({
            "transcript_html":   build_transcript_html(row["text"]),
            "instructions_html": instructions_html,
            "audio_url":         AUDIO_URL_TEMPLATE.format(
                language=quote(language), filename=quote(row["filename"])
            ),
        })
        tasks.append({"data": data})
        tracking_records.append({"task_uid": task_uid, **{c: row.get(c) for c in TRACKING_COLUMNS}})

    if tracking_csv_path:
        write_tracking_csv(tracking_records, tracking_csv_path)
    return tasks


# -----------------------------------------------------------------------------
# HumanAlign / Label Studio API helpers
# -----------------------------------------------------------------------------

def _api_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}


def create_project(
    title: str,
    label_config: str,
    api_key: str,
    base_url: str,
    description: str = "",
) -> int:
    """Create a new Label Studio project and return its integer ID."""
    if not _REQUESTS_AVAILABLE:
        sys.exit("ERROR: 'requests' package is required for API integration. Run: pip install requests")

    resp = _requests.post(
        f"{base_url.rstrip('/')}/api/projects",
        headers=_api_headers(api_key),
        json={"title": title, "label_config": label_config, "description": description},
        timeout=30,
    )
    resp.raise_for_status()
    project_id: int = resp.json()["id"]
    print(f"  Project created: {base_url.rstrip('/')}/projects/{project_id}/")
    return project_id


def import_tasks(project_id: int, tasks: list[dict], api_key: str, base_url: str) -> None:
    """Import a list of task dicts into an existing Label Studio project."""
    if not _REQUESTS_AVAILABLE:
        sys.exit("ERROR: 'requests' package is required for API integration. Run: pip install requests")

    resp = _requests.post(
        f"{base_url.rstrip('/')}/api/projects/{project_id}/import",
        headers=_api_headers(api_key),
        json=tasks,
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()
    n = result.get("task_count") or result.get("added") or len(tasks)
    print(f"  Imported {n} tasks into project {project_id}.")


# -----------------------------------------------------------------------------
# Per-language orchestration
# -----------------------------------------------------------------------------

def process_language(language: str, args: argparse.Namespace) -> None:
    items = load_items(language)
    if items.empty:
        print(f"  WARNING: no items found for {language} — skipping.", file=sys.stderr)
        return

    lang_slug = language.replace(" ", "_")
    output_dir = Path(args.output_dir) / language
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path   = output_dir / f"labeling_config_{lang_slug}.xml"
    tasks_path    = output_dir / f"tasks_{lang_slug}.json"
    tracking_path = output_dir / f"tracking_{lang_slug}.csv"

    label_config = build_labeling_config()
    config_path.write_text(label_config, encoding="utf-8")
    print(f"  Labeling config written to: {config_path}")

    tasks = build_tasks(
        items,
        language=language,
        instructions_html=build_instructions_html(language),
        seed=args.seed,
        tracking_csv_path=tracking_path,
    )
    tasks_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Tasks JSON written to:      {tasks_path}  ({len(tasks)} tasks)")

    if args.api_key:
        project_title = args.project_title or f"Alignment quality — {language}"
        description = (
            f"Forced-alignment quality check for {language} (BibleTTS Sec. 4.3 protocol). "
            f"{len(tasks)} verses. Random seed: {args.seed}."
        )
        project_id = create_project(
            title=project_title,
            label_config=label_config,
            api_key=args.api_key,
            base_url=args.base_url,
            description=description,
        )
        import_tasks(project_id, tasks, api_key=args.api_key, base_url=args.base_url)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Label Studio / HumanAlign tasks for forced-alignment evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--language", choices=LANGUAGES, help="Language to process (or use --all).")
    parser.add_argument("--all", action="store_true", help="Process all languages.")
    parser.add_argument(
        "--output-dir",
        default="humanalign",
        help="Root directory; files go to <output-dir>/<language>/ (default: humanalign).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling tasks (default: 42).")
    parser.add_argument(
        "--api-key",
        help="Label Studio / HumanAlign API token. If provided, creates a project and imports tasks.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"HumanAlign base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--project-title",
        help="Project title when creating via API (default: auto-generated from language).",
    )
    args = parser.parse_args()

    if not args.language and not args.all:
        parser.error("Specify --language <LANG> or --all.")

    for language in LANGUAGES if args.all else [args.language]:
        print(f"\n{'=' * 60}\nProcessing: {language}\n{'=' * 60}")
        process_language(language, args)


if __name__ == "__main__":
    main()
