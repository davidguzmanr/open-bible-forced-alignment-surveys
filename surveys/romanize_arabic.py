"""
Rule-based Latin transcription of fully vowelled (tashkil) Arabic text.

Meant as a reading aid for following the audio, not as a standard
transliteration. It uses a simplified ALA-LC-style scheme: long vowels ā ī ū,
emphatics ḥ ṣ ḍ ṭ ẓ, ʿ for ʿayn, ʾ for hamza (omitted word-initially), shadda
doubles the consonant, and the article is written al- / l- with sun-letter
assimilation (ash-shams). Case endings are kept as written, even though a
reader may drop them before a pause.
"""

import re

CONSONANTS = {
    "ء": "ʾ", "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "ḥ", "خ": "kh",
    "د": "d", "ذ": "dh", "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "ṣ",
    "ض": "ḍ", "ط": "ṭ", "ظ": "ẓ", "ع": "ʿ", "غ": "gh", "ف": "f", "ق": "q",
    "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "و": "w", "ي": "y",
    "ؤ": "ʾ", "ئ": "ʾ",
}
VOWELS = {"َ": "a", "ِ": "i", "ُ": "u",
          "ً": "an", "ٍ": "in", "ٌ": "un"}
SHADDA, SUKUN, TATWEEL = "ّ", "ْ", "ـ"
MARKS = set(VOWELS) | {SHADDA, SUKUN}
PUNCTUATION = {"،": ",", "؟": "?", "؛": ";"}
LONG = {"a": "ā", "i": "ī", "u": "ū"}

# The name of God is usually written without the long-vowel mark; spell it
# out (with an optional prefix) instead of applying the letter rules.
ALLAH_RE = re.compile(r"^(?P<prefix>[وف]?[بل]?)(?P<article>ال)?له$")
ALLAH_PREFIX = {"": "allāh", "و": "wallāh", "ف": "fallāh", "ب": "billāh",
                "ل": "lillāh", "وب": "wabillāh", "ول": "walillāh", "فب": "fabillāh", "فل": "falillāh"}


def _clusters(word: str) -> list[tuple[str, str]]:
    """Split a word into (base character, following diacritics) pairs."""
    clusters: list[tuple[str, str]] = []
    for ch in word:
        if ch == TATWEEL:
            continue
        if ch in MARKS and clusters:
            base, marks = clusters[-1]
            clusters[-1] = (base, marks + ch)
        else:
            clusters.append((ch, ""))
    return clusters


def _vowel(marks: str) -> str:
    return next((VOWELS[m] for m in marks if m in VOWELS), "")


def _lengthen(out: list[str], vowel: str) -> bool:
    """Turn a trailing short vowel into its long form; True if it did."""
    if out and out[-1].endswith(vowel):
        out[-1] = out[-1][:-1] + LONG[vowel]
        return True
    return False


def romanize_word(word: str) -> str:
    clusters = _clusters(word)
    skeleton = "".join(base for base, _ in clusters)
    allah = ALLAH_RE.match(skeleton)
    # Without the article it must be li-llāh (doubled lam), not e.g. lahu "to him".
    if allah and allah.group("prefix") in ALLAH_PREFIX and (
        allah.group("article") or (allah.group("prefix").endswith("ل") and SHADDA in clusters[-2][1])
    ):
        return ALLAH_PREFIX[allah.group("prefix")] + _vowel(clusters[-1][1])
    out: list[str] = []
    sun_article = False  # article lam assimilated into the next consonant

    for i, (base, marks) in enumerate(clusters):
        nxt = clusters[i + 1] if i + 1 < len(clusters) else None
        vowel = _vowel(marks)

        if base == "ا":
            # Definite article: bare alif + unvowelled lam + another letter.
            is_article = (
                not marks and nxt is not None and nxt[0] == "ل"
                and not _vowel(nxt[1]) and i + 2 < len(clusters)
            )
            if is_article or (not marks and nxt is not None and SHADDA in nxt[1]):
                # Hamzat al-wasl (article, alladhī): silent after a prefix.
                out.append("a" if i == 0 else "")
            elif not marks and nxt is None and out and out[-1].endswith(("ū", "w")):
                pass  # silent alif after the plural ending -ū / -aw
            elif vowel:
                out.append(vowel)
            elif not _lengthen(out, "a"):
                out.append("a" if i == 0 else "ā")
        elif base == "ل" and i > 0 and clusters[i - 1] == ("ا", "") and not _vowel(marks) and nxt:
            if SHADDA in nxt[1]:
                sun_article = True
            else:
                out.append("l-")
        elif base in "أإ":
            default = "" if SUKUN in marks else ("i" if base == "إ" else "a")
            out.append(("" if i == 0 else "ʾ") + (vowel or default))
        elif base == "آ":
            out.append(("" if i == 0 else "ʾ") + "ā")
        elif base == "ى":
            if not _lengthen(out, "a"):
                out.append("ā")
        elif base == "ة":
            if vowel:
                out.append("t" + vowel)
            elif not (out and out[-1].endswith("a")):
                out.append("a")
        elif base in "وي" and not marks and _lengthen(out, "u" if base == "و" else "i"):
            pass
        elif base in CONSONANTS:
            c = CONSONANTS[base]
            if SHADDA in marks:
                c = f"{c}-{c}" if sun_article else c + c
            sun_article = False
            out.append(c + vowel)
        else:
            out.append(PUNCTUATION.get(base, base))

    return "".join(out)


# Punctuation that can stick to a word; split off so word-level rules see the
# bare word. (Not \W: Arabic diacritics are not \w and would be stripped too.)
EDGE_PUNCT = "،؟؛.,:;!?\"'«»“”‘’()[]-–—…"


def _romanize_token(token: str) -> str:
    core = token.strip(EDGE_PUNCT)
    if not core:
        return "".join(PUNCTUATION.get(ch, ch) for ch in token)
    start = token.index(core)
    lead, trail = token[:start], token[start + len(core):]
    fix = lambda p: "".join(PUNCTUATION.get(ch, ch) for ch in p)
    return fix(lead) + romanize_word(core) + fix(trail)


def romanize(text: str) -> str:
    return "".join(
        part if part.isspace() else _romanize_token(part)
        for part in re.split(r"(\s+)", text)
    )
