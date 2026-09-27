"""
Local Streamlit version of the HumanSignal alignment-quality task.

Shows the same tasks, in the same order, as humanalign/{lang}/tasks_{lang}.json:
the instructions, the transcript and the verse clip (played from audios/), the
four BibleTTS options, and for a mismatch the required "Where is the problem?"
question and the optional free-text box. One language at a time.

Answers are saved after every submit to
    human-evaluation/annotations-streamlit/{language}.csv
one row per (annotator, task), in the same layout as a Label Studio CSV export
(alignment, location as {"choices": [...]}, words, audio_url, ...), so the file
can go through human-evaluation/anonymize_annotations.py and
analyze_alignment.py like a HumanSignal export.

Usage:
    streamlit run app/annotate.py
"""

import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "surveys"))
from build_survey_alignment import (  # noqa: E402  (options shared with the HumanSignal config)
    ALIGNMENT_CHOICES,
    LANGUAGES,
    LOCATION_CHOICES,
    MISMATCH_CHOICES,
)

OUTPUT_DIR = ROOT / "human-evaluation" / "annotations-streamlit"
COLUMNS = [
    "annotator", "task_uid", "filename", "audio_url", "alignment", "location", "words",
    "lead_time", "created_at", "updated_at",
]


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def tasks_path(language: str) -> Path:
    slug = language.replace(" ", "_")
    return ROOT / "humanalign" / language / f"tasks_{slug}.json"


@st.cache_data
def load_tasks(language: str) -> list[dict]:
    return [t["data"] for t in json.loads(tasks_path(language).read_text(encoding="utf-8"))]


def annotations_path(language: str) -> Path:
    return OUTPUT_DIR / f"{language}.csv"


def load_annotations(language: str) -> pd.DataFrame:
    path = annotations_path(language)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def save_annotation(language: str, record: dict) -> None:
    """Insert or replace the (annotator, task_uid) row; keeps the first created_at."""
    df = load_annotations(language)
    same = (df["annotator"] == record["annotator"]) & (df["task_uid"] == record["task_uid"])
    if same.any():
        record["created_at"] = df.loc[same, "created_at"].iat[0]
    df = pd.concat([df[~same], pd.DataFrame([record])], ignore_index=True) if len(df) else pd.DataFrame([record])
    df = df.sort_values(["annotator", "task_uid"])[COLUMNS]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = annotations_path(language).with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(annotations_path(language))


def saved_answer(done: pd.DataFrame, task_uid: str) -> dict | None:
    row = done[done["task_uid"] == task_uid]
    if row.empty:
        return None
    row = row.iloc[0]
    location = json.loads(row["location"])["choices"] if row["location"] else []
    return {"alignment": row["alignment"], "location": location, "words": row["words"]}


# -----------------------------------------------------------------------------
# Waveform player
# -----------------------------------------------------------------------------

WAVESURFER = "https://cdn.jsdelivr.net/npm/wavesurfer.js@7.12.12/dist"

WAVEFORM_HTML = """
<div style="font-family:sans-serif;color:__TEXT__;">
  <div id="wave"></div>
  <div id="timeline"></div>
  <div style="display:flex;align-items:center;gap:12px;margin-top:8px;font-size:14px;">
    <button id="play" style="padding:4px 14px;border-radius:6px;border:1px solid #888;
            background:transparent;color:inherit;cursor:pointer;min-width:84px;">&#9654; Play</button>
    <span id="time" style="font-variant-numeric:tabular-nums;">0.00 / 0.00 s</span>
    <label style="margin-left:auto;display:flex;align-items:center;gap:6px;">
      Zoom <input id="zoom" type="range" min="0" max="400" value="0">
    </label>
  </div>
  <audio id="fallback" controls style="display:none;width:100%;"></audio>
</div>
<script type="module">
  const src = "__SRC__";
  const fmt = (t) => t.toFixed(2);
  try {
    const { default: WaveSurfer } = await import("__WS__/wavesurfer.esm.js");
    const { default: Timeline } = await import("__WS__/plugins/timeline.esm.js");
    const ws = WaveSurfer.create({
      container: "#wave",
      url: src,
      height: 110,
      waveColor: "#8fa3bf",
      progressColor: "#3b6fd8",
      cursorColor: "#d9534f",
      cursorWidth: 2,
      normalize: true,
      minPxPerSec: 0,
      plugins: [Timeline.create({ container: "#timeline", style: { color: "__TEXT__" } })],
    });
    const play = document.getElementById("play");
    const time = document.getElementById("time");
    const show = () => { time.textContent = `${fmt(ws.getCurrentTime())} / ${fmt(ws.getDuration())} s`; };
    ws.on("ready", show);
    ws.on("timeupdate", show);
    ws.on("play", () => { play.innerHTML = "&#10074;&#10074; Pause"; });
    ws.on("pause", () => { play.innerHTML = "&#9654; Play"; });
    ws.on("finish", () => { play.innerHTML = "&#9654; Play"; });
    play.onclick = () => ws.playPause();
    document.getElementById("zoom").oninput = (e) => ws.zoom(Number(e.target.value));
  } catch (err) {
    // CDN unreachable: fall back to a plain player so the task can still be done.
    document.getElementById("wave").parentElement.querySelectorAll("div").forEach((d) => d.remove());
    const audio = document.getElementById("fallback");
    audio.src = src;
    audio.style.display = "block";
  }
</script>
"""


def waveform_player(audio_file: Path, fallback_url: str) -> None:
    """wavesurfer.js waveform (as in HumanSignal) with play/pause, click-to-seek and zoom."""
    if audio_file.exists():
        src = "data:audio/wav;base64," + base64.b64encode(audio_file.read_bytes()).decode()
    else:
        src = fallback_url
    dark = st.context.theme.type == "dark"
    html = (
        WAVEFORM_HTML.replace("__SRC__", src)
        .replace("__WS__", WAVESURFER)
        .replace("__TEXT__", "#e6e6e6" if dark else "#262730")
    )
    st.iframe(html, height=195)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Alignment annotation", page_icon="🎧", layout="centered")

available = [lang for lang in LANGUAGES if tasks_path(lang).exists()]
with st.sidebar:
    st.header("Alignment annotation")
    language = st.selectbox("Language", available)
    annotator = st.text_input("Annotator name", key="annotator").strip()

if not annotator:
    st.info("Enter your name in the sidebar to start.")
    st.stop()

tasks = load_tasks(language)
done = load_annotations(language)
done = done[done["annotator"] == annotator]
labeled = set(done["task_uid"])

# Current task index, per language and annotator. Starts at the first unlabeled task.
idx_key = f"idx|{language}|{annotator}"
if idx_key not in st.session_state:
    st.session_state[idx_key] = next(
        (i for i, t in enumerate(tasks) if t["task_uid"] not in labeled), 0
    )


def go_to(i: int) -> None:
    st.session_state[idx_key] = max(0, min(len(tasks) - 1, i))


with st.sidebar:
    st.progress(len(labeled) / len(tasks), text=f"{len(labeled)} / {len(tasks)} labeled")
    st.selectbox(
        "Go to task",
        range(len(tasks)),
        index=st.session_state[idx_key],
        format_func=lambda i: f"{i + 1}  {'✓' if tasks[i]['task_uid'] in labeled else '·'}",
        key=f"jump|{language}|{annotator}|{st.session_state[idx_key]}",
        on_change=lambda k: go_to(st.session_state[k]),
        args=(f"jump|{language}|{annotator}|{st.session_state[idx_key]}",),
    )
    remaining = [i for i, t in enumerate(tasks) if t["task_uid"] not in labeled]
    if remaining:
        st.button("First unlabeled task", on_click=go_to, args=(remaining[0],))
    else:
        st.success("All tasks labeled.")
    show_latin = False
    if any("transcript_latin" in t for t in tasks):
        show_latin = st.toggle("Show Latin transcription", value=False)
    st.caption(f"Saving to `{annotations_path(language).relative_to(ROOT)}`")

idx = st.session_state[idx_key]
task = tasks[idx]
uid = task["task_uid"]
previous = saved_answer(done, uid)

# Lead time: seconds from first showing this task to submitting it.
shown_key = f"shown|{language}|{annotator}|{uid}"
st.session_state.setdefault(shown_key, time.time())

st.markdown(task["instructions_html"], unsafe_allow_html=True)
st.caption(f"Task {idx + 1} of {len(tasks)}" + ("  ·  already labeled" if previous else ""))

with st.container(border=True):
    st.markdown("**Transcript:**")
    # text-align:start is resolved per element, so right-to-left scripts (dir="auto"
    # in the task HTML) align right despite Streamlit's default left alignment.
    st.markdown(f'<div style="text-align:start">{task["transcript_html"]}</div>', unsafe_allow_html=True)
    if show_latin and task.get("transcript_latin"):
        st.caption(task["transcript_latin"])

audio_file = ROOT / "audios" / language / task["filename"]
waveform_player(audio_file, task["audio_url"])

# Widget keys include the task, so every task starts from its saved answer (or empty).
wkey = f"{language}|{annotator}|{uid}"
alignment = st.radio(
    "**Which option best describes the audio?**",
    ALIGNMENT_CHOICES,
    index=ALIGNMENT_CHOICES.index(previous["alignment"]) if previous else None,
    key=f"alignment|{wkey}",
)

location: list[str] = []
words = ""
if alignment in MISMATCH_CHOICES:
    st.markdown("**Where is the problem?** (select all that apply)")
    for col, choice in zip(st.columns(len(LOCATION_CHOICES)), LOCATION_CHOICES):
        with col:
            if st.checkbox(
                choice,
                value=bool(previous) and choice in previous["location"],
                key=f"location|{choice}|{wkey}",
            ):
                location.append(choice)
    words = st.text_area(
        "**Which words are extra or missing?** (optional)",
        value=previous["words"] if previous else "",
        placeholder="e.g. extra: 'Chapter 5' at the start",
        height=80,
        key=f"words|{wkey}",
    )

left, middle, right = st.columns([1, 2, 1])
with left:
    st.button("← Previous", on_click=go_to, args=(idx - 1,), disabled=idx == 0, use_container_width=True)
with right:
    st.button("Next →", on_click=go_to, args=(idx + 1,), disabled=idx == len(tasks) - 1,
              use_container_width=True)
with middle:
    submitted = st.button("Update" if previous else "Submit", type="primary", use_container_width=True)

if submitted:
    if alignment is None:
        st.error("Please choose one option.")
    elif alignment in MISMATCH_CHOICES and not location:
        st.error("Please select where the extra or missing words are.")
    else:
        now = datetime.now(timezone.utc).isoformat()
        save_annotation(language, {
            "annotator": annotator,
            "task_uid": uid,
            "filename": task["filename"],
            "audio_url": task["audio_url"],
            "alignment": alignment,
            "location": json.dumps({"choices": location}, ensure_ascii=False) if location else "",
            "words": words.strip(),
            "lead_time": round(time.time() - st.session_state[shown_key], 3),
            "created_at": now,
            "updated_at": now,
        })
        # Continue with the next unlabeled task after this one, else the next task.
        labeled.add(uid)
        following = [i for i in range(idx + 1, len(tasks)) if tasks[i]["task_uid"] not in labeled]
        go_to(following[0] if following else idx + 1)
        st.rerun()
