# open-bible-forced-alignment-surveys

Human evaluation of the forced-alignment quality of [davidguzmanr/open-bible-resources](https://huggingface.co/datasets/davidguzmanr/open-bible-resources).

## Overview

26 Open.Bible languages ship Biblica timing files that give verse boundaries. The other **11 languages** were segmented into verses with zero-shot forced alignment ([ReadAlongs](https://github.com/ReadAlongs/Studio), see [open-bible-resources](https://github.com/davidguzmanr/open-bible-resources)), so there is no reference to score them against. Instead, native speakers judge whether each aligned `<audio, transcript>` pair matches, following Section 4.3 of [BibleTTS (Meyer et al., Interspeech 2022)](https://arxiv.org/abs/2207.03546).

- **Languages:** Arabic Standard, Chichewa, Dawro, Gamo, Gofa, Haitian Creole, Kikuyu, Luganda, Shona, Swahili, Turkish
- **Samples:** 50 verses per language from the `train` split (the data the TTS models were trained on)
- **Annotators:** 3 per language
- **Platform:** [HumanSignal / Label Studio](https://app.humansignal.com), one project per language

### Task

Each task shows one verse transcript and its audio clip. Annotators choose the one option that best describes the pair (wording from BibleTTS, with "(exact match)" added; the paper lists that option last, here it comes first):

1. No missing or extra words (exact match)
2. Audio contains EXTRA words not in the transcript
3. Audio is MISSING words that are in the transcript
4. Audio is MISSING words AND includes EXTRA words

For options 2–4, two follow-ups appear: **where** the problem is (start / middle / end of the clip, select all that apply; required) and a free-text box for **which words** are extra or missing (optional).

### Sampling

Every released verse already passed the outlier filter in open-bible-resources, which drops verses whose seconds-per-character ratio is more than 3 standard deviations from the language mean. To survey well-aligned data rather than borderline cuts, we keep the **20% of `train` verses closest to the mean speaking rate** (smallest |z| of that same ratio) and draw 50 verses at random from that pool (seed 42). Results therefore describe the typical part of each corpus, not a uniform sample of it, which is how this differs from the paper's fully random sample. Verses already used as reference recordings in the [TTS listening test](https://github.com/davidguzmanr/open-bible-surveys) are excluded.

Each verse is tagged from the source USX text with the situations where forced alignment is most likely to fail: `is_first_verse` (after the spoken chapter announcement), `heading_before` / `heading_after` (a section heading or Psalm title is next to the verse; headings are not in the aligned text, so if they are read aloud they leak into a neighbouring clip) and `is_verse_range` (merged verses such as 3–4).

## Structure

```
data/{language}.csv                 sampled verses: text, metadata, speaking-rate z-score, tags
audios/{language}/*.wav             original 22.05 kHz clips from the dataset
humanalign/{language}/
    labeling_config_{lang}.xml      paste into the Label Studio labeling config
    tasks_{lang}.json               import into the Label Studio project
    tracking_{lang}.csv             task_uid -> filename, metadata and tags
human-evaluation/
    annotations/                    raw Label Studio CSV exports (git-ignored: contain emails)
    annotations-anonymized/         anonymized, deduplicated exports
    results/                        analysis outputs
```

## Generating the surveys

```bash
pip install -r requirements.txt

# 1. Sample 50 verses per language and save their audio (reads the HF dataset,
#    from the local cache if it has been downloaded already)
python scripts/sample_verses.py --all

# 2. Build the Label Studio config, tasks and tracking CSV
python surveys/build_survey_alignment.py --all
```

Both scripts also accept `--language <LANG>`. Arabic Standard tasks also carry a `transcript_latin` field (rule-based romanization from `surveys/romanize_arabic.py`) to help follow the audio; the labeling config does not display it, so annotators only see the Arabic script. Audio is served to Label Studio from this repository's `raw.githubusercontent.com` URLs, so **commit and push `audios/` before importing the tasks**.

## Uploading to HumanAlign

### Step 1: Create a project

Log into [app.humansignal.com](https://app.humansignal.com) → **Create Project** → give it a name like `Alignment quality — Swahili`.

### Step 2: Set the labeling config

Go to **Project Settings → Labeling Interface** → open the code editor (`</>` or **Custom template**) → paste the contents of `labeling_config_{lang}.xml` → **Save**. Check in the preview that the "Where is the problem?" block appears only after choosing one of the last three options.

### Step 3: Import tasks

Go to the project's **Data Manager** → **Import** → upload `tasks_{lang}.json`. You should see 50 tasks with `transcript`, `audio_url` and `filename` fields.

### Step 4: Set task assignment to manual

Go to **Project Settings → Annotation → Task Assignment** → set to **Manual**, so annotators pick up tasks from the queue freely.

### Step 5: Configure overlap

Go to **Project Settings → Quality** → set **Annotations per task** to at least **3**, so every annotator can label every task.

### Step 6: Add annotators

Go to **Members** → invite the 3 annotators by email → set their role to **Annotator**.

Steps 1–3 can also be done from the command line with `--api-key YOUR_TOKEN` (and optionally `--base-url`, `--project-title`).

## Analysis

1. Export each project as CSV into `human-evaluation/annotations/{language}.csv`.
2. Anonymize annotators and drop duplicate submissions:
   ```bash
   python human-evaluation/anonymize_annotations.py
   ```
3. Summarise:
   ```bash
   python human-evaluation/analyze_alignment.py
   ```

Each verse gets the majority label of its annotators. If no label has a strict plurality (e.g. three different answers), it is counted as **Conflict**, as in the paper. The script reports the Table 3 breakdown (EM / Add. / Miss. / Both / Conflict, % of verses) per language, Krippendorff's alpha for inter-annotator agreement, the same breakdown by risk tag, and counts of the "where" answers.
