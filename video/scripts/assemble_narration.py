"""Assemble the narration from the narrator's takes in `video/recordings/*.m4a` (git-ignored).

Each piece is cut at a silence, brought to a common loudness, and joined with short fades, so
`timeline.py narration` can align the result like a single take. Pauses longer than LONG_PAUSE_S are
shortened to KEEP_PAUSE_S, and an optional small tempo change (pitch kept) can bring the cut under 3:00.

How to fill PIECES once the recording arrives (do not guess cut points):
1. Put the takes in `video/recordings/` (for example `take-01.m4a`, `pickups-01.m4a`).
2. Find each sentence roughly: `whisper-cli -m video/.models/ggml-large-v3-turbo-q5_0.bin -f <take as 16 kHz wav> -ml 1 -sow -ojf`.
   Whisper's word times drift by up to 2 s, so use them only to locate a sentence.
3. Place every cut in a silence found by an energy scan of the take (run `python3 video/scripts/assemble_narration.py --silences <take>`),
   and write one entry per kept stretch: (file name in recordings/, start_s, end_s, gap_after_s, label).
   Cut stumbles and restarts; keep one clean read of every sentence of narration.md, in order.
4. Run this script, then `python3 video/scripts/timeline.py narration video/public/audio/narration-assembled.wav`
   and read its "extra words heard" line: anything listed there is an ad-lib or a stumble still in the cut.

Usage, from the repository root:
    python3 video/scripts/assemble_narration.py                    # build narration-assembled.wav from PIECES
    python3 video/scripts/assemble_narration.py --silences FILE     # list the silences of one take
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDINGS = ROOT / "video/recordings"
OUT = ROOT / "video/public/audio/narration-assembled.wav"
RATE = 48_000
TARGET_LUFS = -16.6
FADE_S = 0.03
SENTENCE_GAP_S = 0.45
PARAGRAPH_GAP_S = 0.8
LONG_PAUSE_S = 1.4
KEEP_PAUSE_S = 1.0  # narration.md: "/" is a pause of about one second
TEMPO = 1.0  # raise to 1.02 or 1.03 only if the cut runs over 2:59

# (file in video/recordings/, start_s, end_s, gap_after_s, label). Empty until the recording arrives;
# every entry gets a comment saying why the cut sits where it does.
PIECES: list[tuple[str, float, float, float, str]] = [
]


def _loudness(path: Path) -> float:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    blob = out[out.rindex("{"): out.rindex("}") + 1]
    return float(json.loads(blob)["input_i"])


def _levels(source: Path):
    import numpy as np

    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-"],
                         capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    hop = RATE // 100  # 10 ms
    frames = len(audio) // hop
    level = 20 * np.log10(np.sqrt((audio[:frames * hop].reshape(frames, hop) ** 2).mean(1)) + 1e-9)
    return audio, hop, frames, level < np.percentile(level, 95) - 38


def silences(source: Path, minimum_s: float = 0.25) -> list[tuple[float, float]]:
    """Stretches quieter than 38 dB below the take's loud parts, at least `minimum_s` long."""
    _, _, frames, quiet = _levels(source)
    out, start = [], None
    for index in range(frames + 1):
        silent = index < frames and quiet[index]
        if silent and start is None:
            start = index
        elif not silent and start is not None:
            if (index - start) / 100 >= minimum_s:
                out.append((start / 100, index / 100))
            start = None
    return out


def _shorten_pauses(source: Path, target: Path) -> float:
    """Shorten every pause longer than LONG_PAUSE_S to KEEP_PAUSE_S, cutting from its middle."""
    import numpy as np

    audio, hop, frames, quiet = _levels(source)
    keep = np.ones(len(audio), dtype=bool)
    removed = 0.0
    start = None
    for index in range(frames + 1):
        silent = index < frames and quiet[index]
        if silent and start is None:
            start = index
        elif not silent and start is not None:
            length = (index - start) / 100
            if length > LONG_PAUSE_S:
                cut = length - KEEP_PAUSE_S
                middle = start * hop + int((length / 2) * RATE)
                half = int(cut / 2 * RATE)
                keep[middle - half: middle + half] = False
                removed += cut
            start = None
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-i", "-", str(target)],
                   input=audio[keep].tobytes(), check=True)
    return removed


def main() -> None:
    if not PIECES:
        raise SystemExit("PIECES is empty: fill it from the takes in video/recordings/ (see the instructions at the top of this file)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        parts: list[Path] = []
        clock = 0.0
        for index, (name, start, end, gap, label) in enumerate(PIECES):
            source = RECORDINGS / name
            raw = tmpdir / f"{index:02d}-raw.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-to", str(end), "-i", str(source), "-vn",
                            "-ac", "1", "-ar", str(RATE), "-af", "highpass=f=80", str(raw)], check=True)
            gain = TARGET_LUFS - _loudness(raw)
            length = end - start
            piece = tmpdir / f"{index:02d}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af",
                            f"volume={gain:.2f}dB,afade=t=in:d={FADE_S},afade=t=out:st={length - FADE_S:.3f}:d={FADE_S}"
                            + (f",apad=pad_dur={gap}" if gap else ""), "-ar", str(RATE), str(piece)], check=True)
            print(f"{clock:7.2f}s  {length:6.2f}s  gain {gain:+5.1f} dB  {label}")
            clock += length + gap
            parts.append(piece)
        listing = tmpdir / "parts.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        joined = tmpdir / "joined.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)], check=True)
        tightened = tmpdir / "tightened.wav"
        removed = _shorten_pauses(joined, tightened)
        print(f"pauses: {removed:.1f}s removed from pauses longer than {LONG_PAUSE_S}s")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tightened), "-af", f"atempo={TEMPO}", "-ar", str(RATE), str(OUT)], check=True)
    total = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(OUT)],
                                 capture_output=True, text=True, check=True).stdout)
    print(f"{OUT.relative_to(ROOT)}: {clock:.1f}s joined, {total:.1f}s at tempo {TEMPO}"
          " (the video adds a 2 s lead-in and a 4 s end hold)")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--silences":
        for a, b in silences(Path(sys.argv[2])):
            print(f"{a:8.2f} {b:8.2f}  ({b - a:.2f}s)")
    else:
        main()
