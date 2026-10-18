"""Build the video's timeline (one beat per narration paragraph) and its voice track.

Two sources, one output format (`video/src/content/timeline.json` plus a WAV in `video/public/audio/`):

- `scratch`: synthesizes each paragraph of the narration with the macOS `say` voice, only to time
  the cut before the narrator records. It is a placeholder and is labelled as one (the video stamps
  every frame "DRAFT, placeholder voice" while `source` starts with "scratch"). When whisper.cpp and
  a model are present, the scratch voice is also transcribed so captions and cues get word times.
- `narration`: takes the narrator's own recording, transcribes it with whisper.cpp for word
  timings, and finds where each paragraph starts and ends.

The narration text is read from `video/narration.md`, from the first heading that starts with
`## The narration (` to the next `## ` heading: paragraphs are blocks of `>` lines separated by an
empty `>` line, and `/` marks a pause.

Usage, from the repository root:
    python3 video/scripts/timeline.py scratch [WPM]
    python3 video/scripts/timeline.py narration video/public/audio/narration-assembled.wav
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "video/narration.md"
DATA = ROOT / "video/src/content/timeline.json"
AUDIO_DIR = ROOT / "video/public/audio"
BEATS = [
    "cold_open", "intro", "walk_graph", "benchmark", "arms", "results", "audit", "falkordb",
    "check_it", "close",
]
HEADING = "## The narration ("
LEAD_IN_S = 2.0  # the cold open's records start stacking before the first word
GAP_S = 0.5
END_HOLD_S = 4.0  # the close card stays up for its own sentence plus this hold
PAUSE_MS = 1000  # narration.md: "/" is a pause of about one second
SAMPLE_RATE = 48_000
LIMIT_S = 179.0
# Scratch only: extra silence after a beat whose picture needs reading time after its last word
# (the schema, the Alice Davis rows, the never-seen split, the passing audit round, the what-if items).
# A recording keeps the narrator's own pause between paragraphs instead (narration.md asks for two seconds).
EXTRA_AFTER_S = {"intro": 0.8, "walk_graph": 0.6, "results": 1.5, "audit": 0.8, "falkordb": 1.2, "check_it": 0.5}
# The placeholder voice reads a few words better spelled the way the narrator will say them
# (narration.md, "How to say it"). Only the synthesized audio changes; captions keep the script.
SAY_AS = {"GRAPH.COPY": "graph copy", "Qwen3": "Chwen three", "Hi, I'm Zeyu.": "Hi, I'm Zay-yoo."}


# Recording only: where the narrator's words differ from the script and both whisper models (large-v3-turbo
# and small.en) hear the same different word, the captions follow what he said. (beat, written, said)
AS_SPOKEN = [
    ("intro", "Hi, I'm Zeyu.", "Hey, I'm Zeyu."),
    ("benchmark", "Salesforce's HERB benchmark", "Salesforce HERB benchmark"),
    ("results", "twenty-six, with twelve times", "twenty-six, and twelve times"),
]


def paragraphs() -> list[str]:
    text = SCRIPT.read_text(encoding="utf-8")
    start = next((m.start() for m in re.finditer(r"^## The narration \(", text, re.M)), None)
    if start is None:
        raise SystemExit(f"no heading starting with {HEADING!r} in {SCRIPT}")
    following = re.search(r"^## ", text[start + 3:], re.M)
    end = start + 3 + following.start() if following else len(text)
    blocks: list[list[str]] = [[]]
    for line in text[start:end].splitlines():
        if not line.startswith(">"):
            continue
        body = line[1:].strip()
        if body:
            blocks[-1].append(body)
        elif blocks[-1]:
            blocks.append([])
    found = [" ".join(block) for block in blocks if block]
    if len(found) != len(BEATS):
        raise SystemExit(f"expected {len(BEATS)} narration paragraphs, found {len(found)}")
    return found


def display_text(text: str) -> str:
    """The paragraph as captioned: the pause marks removed."""
    return re.sub(r"\s+", " ", text.replace(" /", "").replace("/", "")).strip()


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(out.strip())


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
# The larger model hears an accented narrator far better (88% to over 95% of words matched); the small
# one is the fallback when the large file has not been downloaded.
_LARGE = ROOT / "video/.models/ggml-large-v3-turbo-q5_0.bin"
MODEL = _LARGE if _LARGE.exists() else ROOT / "video/.models/ggml-small.en.bin"
# Proper nouns whisper would otherwise mishear.
VOCABULARY = ("Hi, I'm Zeyu. inherit-brain, FalkorDB, HERB, Qwen, Qwen3, Nebius, GraphRAG SDK, Cypher, "
              "Alice Davis, Julia Smith, GRAPH.COPY, SupportForce, Salesforce.")


def _number_words(n: int) -> list[str]:
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        return [_TENS[n // 10]] + ([_ONES[n % 10]] if n % 10 else [])
    if n == 100:
        return ["hundred"]
    return [str(n)]


def _tokens(text: str) -> list[str]:
    """Lower-case words for alignment: punctuation dropped, hyphens and dots split, digits spelled out."""
    out: list[str] = []
    for raw in re.split(r"[\s/\-.]+", text.lower()):
        for part in re.findall(r"[a-z']+|[0-9]+", raw):
            word = part.strip("'")
            if not word:
                continue
            if word.isdigit():
                out.extend(_number_words(int(word)))
            else:
                out.append(word)
    return out


def _transcribe(wav16k: Path, workdir: Path) -> list[tuple[str, float, float]]:
    subprocess.run(["whisper-cli", "-m", str(MODEL), "-f", str(wav16k), "-ml", "1", "-sow",
                    "--prompt", VOCABULARY, "-ojf", "-of", str(workdir / "words"), "-np"],
                   check=True, capture_output=True)
    data = json.loads((workdir / "words.json").read_text())
    words = []
    for segment in data["transcription"]:
        text = segment["text"].strip()
        if not text:
            continue
        words.append((text, segment["offsets"]["from"] / 1000, segment["offsets"]["to"] / 1000))
    return words


def _align(wav: Path, texts: list[str], workdir: Path):
    """Transcribe `wav` and map every script token to a heard token where the two sequences agree."""
    import difflib

    wav16k = workdir / "16k.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-ac", "1", "-ar", "16000", str(wav16k)], check=True)
    heard: list[tuple[str, float, float]] = []
    for text, start, end in _transcribe(wav16k, workdir):
        for token in _tokens(text):
            heard.append((token, start, end))
    script_tokens: list[tuple[int, str]] = []
    for index, text in enumerate(texts):
        script_tokens.extend((index, token) for token in _tokens(text))
    matcher = difflib.SequenceMatcher(a=[t for _, t in script_tokens], b=[t for t, _, _ in heard], autojunk=False)
    mapped: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            mapped[block.a + k] = block.b + k
    return script_tokens, mapped, heard


def _display_words(text: str, beat_index: int, script_tokens: list[tuple[int, str]], mapped: dict[int, int],
                   heard: list[tuple[str, float, float]], shift: float, start: float, end: float) -> list[dict]:
    """Every word of the script as written (case, punctuation, digits kept, for the captions), timed.

    A word takes the span of its heard tokens, clamped to its beat; a word the transcription missed
    (an accent, a name) is placed evenly in the gap between its timed neighbours, so captions and cues
    never drop it."""

    first = next(i for i, (p, _) in enumerate(script_tokens) if p == beat_index)
    cursor = first
    words: list[dict] = []
    for display in text.split():
        count = len(_tokens(display))
        hits = [mapped[i] for i in range(cursor, cursor + count) if i in mapped]
        cursor += count
        if hits:
            s = min(max(heard[min(hits)][1] + shift, start), end)
            e = min(max(heard[max(hits)][2] + shift, s), end)
            words.append({"w": display, "start_s": s, "end_s": e})
        else:
            words.append({"w": display, "start_s": None, "end_s": None})
    # Times must not run backwards: a word heard out of order is treated as missed.
    last = start
    for w in words:
        if w["start_s"] is not None and w["start_s"] < last:
            w["start_s"] = w["end_s"] = None
        elif w["start_s"] is not None:
            last = w["start_s"]
    i = 0
    while i < len(words):
        if words[i]["start_s"] is not None:
            i += 1
            continue
        j = i
        while j < len(words) and words[j]["start_s"] is None:
            j += 1
        left = words[i - 1]["end_s"] if i > 0 else start
        right = words[j]["start_s"] if j < len(words) else end
        step = max(right - left, 0.0) / (j - i)
        for k in range(i, j):
            words[k]["start_s"] = left + step * (k - i)
            words[k]["end_s"] = left + step * (k - i + 1)
        i = j
    return [{"w": w["w"], "start_s": round(w["start_s"], 3), "end_s": round(w["end_s"], 3)} for w in words]


def _whisper_ready() -> bool:
    return shutil.which("whisper-cli") is not None and MODEL.exists()


def scratch(rate_wpm: int = 150) -> None:
    texts = paragraphs()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    beats = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        parts: list[Path] = []
        silence = tmpdir / "gap.wav"
        lead = tmpdir / "lead.wav"
        tail = tmpdir / "tail.wav"
        for name, seconds in ((silence, GAP_S), (lead, LEAD_IN_S), (tail, END_HOLD_S)):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                            f"anullsrc=r={SAMPLE_RATE}:cl=mono", "-t", str(seconds), str(name)], check=True)
        parts.append(lead)
        clock = LEAD_IN_S
        for index, (beat, text) in enumerate(zip(BEATS, texts)):
            spoken = text
            for written, said in SAY_AS.items():
                spoken = spoken.replace(written, said)
            spoken = spoken.replace("/", f"[[slnc {PAUSE_MS}]]")
            aiff = tmpdir / f"{index:02d}.aiff"
            wav = tmpdir / f"{index:02d}.wav"
            subprocess.run(["say", "-v", "Samantha", "-r", str(rate_wpm), "-o", str(aiff), spoken], check=True)
            # `say` ends each file with a trailing silence; trim it so beat ends sit on the last word.
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(aiff), "-ar", str(SAMPLE_RATE), "-ac", "1",
                            "-af", "areverse,silenceremove=start_periods=1:start_threshold=-50dB,areverse",
                            str(wav)], check=True)
            length = _duration(wav)
            beats.append({"id": beat, "text": display_text(text), "start_s": round(clock, 3), "end_s": round(clock + length, 3)})
            clock += length
            parts.append(wav)
            if index < len(texts) - 1:
                parts.append(silence)
                clock += GAP_S
                extra = EXTRA_AFTER_S.get(beat, 0.0)
                if extra:
                    hold = tmpdir / f"{index:02d}-hold.wav"
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
                                    "-t", str(extra), str(hold)], check=True)
                    parts.append(hold)
                    clock += extra
        parts.append(tail)
        clock += END_HOLD_S
        listing = tmpdir / "parts.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        out = AUDIO_DIR / "scratch-voice.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)], check=True)
        coverage = None
        if _whisper_ready():
            # The beat boundaries are exact (each paragraph is its own file); whisper only places the words.
            script_tokens, mapped, heard = _align(out, [b["text"] for b in beats], tmpdir)
            for index, beat in enumerate(beats):
                beat["words"] = _display_words(beat["text"], index, script_tokens, mapped, heard, 0.0,
                                               beat["start_s"], beat["end_s"])
            coverage = len(mapped) / max(1, len(script_tokens))
    timeline = {
        "source": "scratch: macOS Samantha voice, a placeholder to time the cut; the final video uses the narrator's recording",
        "voice": "audio/scratch-voice.wav",
        "fps": 30,
        "lead_in_s": LEAD_IN_S,
        "end_hold_s": END_HOLD_S,
        "total_s": round(clock, 3),
        **({"alignment": {"script_words_matched": round(coverage, 3)}} if coverage is not None else {}),
        "beats": beats,
    }
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
    words = sum(len(b["text"].split()) for b in beats)
    matched = f", {coverage:.0%} of words placed by whisper" if coverage is not None else ", no word times (whisper missing)"
    print(f"scratch voice {clock:.1f}s over {len(beats)} beats, {words} words at -r {rate_wpm}{matched} -> {DATA.relative_to(ROOT)}")
    pauses = texts_pauses(texts)
    print(f"at 150 words per minute the same script would run about {words / 150 * 60 + pauses + LEAD_IN_S + END_HOLD_S + GAP_S * 9:.0f}s"
          f" ({words / 150 * 60:.0f}s of words, {pauses:.0f} one-second pauses, lead-in, gaps and end hold)")
    if clock > LIMIT_S:
        print("WARNING: longer than 2:59; raise the rate or trim before rendering")


def texts_pauses(texts: list[str]) -> float:
    return sum(t.count("/") for t in texts) * 1.0


def narration(recording: Path, already_clean: bool = False) -> None:
    """Time the beats to the narrator's own recording, cleaned and loudness-normalised.

    `already_clean`: the recording comes from assemble_narration.py, which has denoised it and
    normalised it to -16 LUFS; only the leading silence is trimmed, so it is not processed twice."""
    texts = paragraphs()
    for beat, written, said in AS_SPOKEN:
        index = BEATS.index(beat)
        if written not in texts[index]:
            raise SystemExit(f"AS_SPOKEN: {written!r} is not in the {beat} paragraph")
        texts[index] = texts[index].replace(written, said)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cleaned = tmpdir / "cleaned.wav"
        # High-pass rumble, light FFT denoise, then broadcast loudness (-16 LUFS, -1.5 dBTP); an assembled
        # take only loses its leading silence.
        chain = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.15"
        if not already_clean:
            chain += ",highpass=f=80,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(recording), "-ac", "1", "-ar", str(SAMPLE_RATE),
                        "-af", chain, str(cleaned)], check=True)
        script_tokens, mapped, heard = _align(cleaned, texts, tmpdir)
        matched_heard = set(mapped.values())
        # Lead-in: the cold open needs LEAD_IN_S before the first word.
        first_word = heard[min(mapped.values())][1] if mapped else 0.0
        shift = max(0.0, LEAD_IN_S - first_word)
        beats = []
        for index, beat in enumerate(BEATS):
            positions = [i for i, (p, _) in enumerate(script_tokens) if p == index and i in mapped]
            if not positions:
                raise SystemExit(f"could not find beat {beat} in the recording; check the take")
            start = heard[mapped[positions[0]]][1] + shift
            end = heard[mapped[positions[-1]]][2] + shift
            text = display_text(texts[index])
            beats.append({"id": beat, "text": text, "start_s": round(start, 3), "end_s": round(end, 3),
                          "words": _display_words(text, index, script_tokens, mapped, heard, shift, start, end)})
        extra = [(t, round(s + shift, 2), round(e + shift, 2)) for i, (t, s, e) in enumerate(heard) if i not in matched_heard]
        # Script words with no heard counterpart, placed after the last matched word before them.
        missing = []
        for i, (beat_index, token) in enumerate(script_tokens):
            if i in mapped:
                continue
            before = [mapped[j] for j in range(i) if j in mapped]
            missing.append((token, BEATS[beat_index], round(heard[before[-1]][2] + shift, 2) if before else 0.0))
        voice_end = _duration(cleaned) + shift
        total = max(voice_end, beats[-1]["end_s"]) + END_HOLD_S
        out = AUDIO_DIR / "narration.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(cleaned), "-af", f"adelay={int(shift * 1000)}",
                        "-ar", str(SAMPLE_RATE), "-ac", "1", str(out)], check=True)
    coverage = len(mapped) / max(1, len(script_tokens))
    timeline = {
        "source": f"narration: the narrator's recording {recording.name}, cleaned and aligned with whisper.cpp",
        "voice": "audio/narration.wav",
        "fps": 30,
        "lead_in_s": round(beats[0]["start_s"], 3),
        "end_hold_s": END_HOLD_S,
        "total_s": round(total, 3),
        "alignment": {"script_words_matched": round(coverage, 3), "unmatched_heard_words": extra, "unmatched_script_words": missing},
        "beats": beats,
    }
    DATA.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
    print(f"narration {total:.1f}s, {coverage:.0%} of script words matched, {len(extra)} extra words heard -> {DATA.relative_to(ROOT)}")
    if extra:
        print("extra words heard (ad-libs or stumbles to cut in assemble_narration.py):",
              " ".join(f"{t}@{s}" for t, s, _ in extra[:60]))
    if missing:
        print("script words not heard (skipped, changed or misheard):", " ".join(f"{t}@{s}" for t, _, s in missing[:60]))
    if total > LIMIT_S:
        print("WARNING: longer than 2:59; trim a pause or a sentence before rendering")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "scratch":
        rate = int(sys.argv[2]) if len(sys.argv) > 2 else 150
        scratch(rate)
    elif mode == "narration" and len(sys.argv) > 2:
        narration(Path(sys.argv[2]), already_clean="--clean" in sys.argv[3:])
    else:
        raise SystemExit("usage: timeline.py scratch [WPM] | timeline.py narration RECORDING [--clean]")
