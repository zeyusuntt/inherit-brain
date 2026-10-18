"""Write the video's captions as an SRT file for YouTube, from the same word timings the burned-in captions use.

Chunks break at a sentence end or after 12 words, and a caption stays up through a pause of up to
1 s, as on screen (`captionChunks` in src/timing.ts).

Usage, from the repository root:
    python3 video/scripts/make_srt.py video/out/inherit-brain.srt
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIMELINE = ROOT / "video/src/content/timeline.json"
MAX_WORDS = 12
HOLD_S = 1.0


def _stamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def chunks() -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
    for beat in json.loads(TIMELINE.read_text())["beats"]:
        words = beat["words"]
        first = 0
        beat_chunks = []
        for index, word in enumerate(words):
            sentence_end = re.search(r"[.!?][\"'’”]?$", word["w"]) is not None
            if sentence_end or index - first + 1 >= MAX_WORDS or index == len(words) - 1:
                beat_chunks.append([words[first]["start_s"], words[index]["end_s"],
                                    " ".join(w["w"] for w in words[first:index + 1])])
                first = index + 1
        for current, following in zip(beat_chunks, beat_chunks[1:]):
            if following[0] - current[1] <= HOLD_S:
                current[1] = following[0]
        out.extend((start, end, text) for start, end, text in beat_chunks)
    return out


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "video/out/inherit-brain.srt"
    lines = []
    for number, (start, end, text) in enumerate(chunks(), start=1):
        lines += [str(number), f"{_stamp(start)} --> {_stamp(max(end, start + 0.5))}", text, ""]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"{target}: {number} captions")


if __name__ == "__main__":
    main()
