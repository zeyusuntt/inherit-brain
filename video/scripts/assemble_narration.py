"""Assemble the narration from the narrator's takes in `video/recordings/` (git-ignored).

Each take is first cleaned as a whole: a high-pass at 80 Hz, an FFT denoiser (ffmpeg `afftdn`) that
learns its noise profile from a pause of room noise in that take, and a gentle gate that only lowers
the pauses. Pieces are then cut from the cleaned take at silences, joined with 30 ms fades and set
gaps, pauses longer than LONG_PAUSE_S are shortened to KEEP_PAUSE_S, an optional small tempo change
(pitch kept, never above 1.04) brings the cut under 3:00, and the result is normalised to -16 LUFS.
`timeline.py narration --clean` then aligns it like a single take, without denoising it again.

How to fill PIECES when a recording arrives (do not guess cut points):
1. Put the takes in `video/recordings/` and convert each to mono 48 kHz WAV
   (an Apple voice memo: `ffmpeg -i take.qta -map 0:0 -ac 1 -ar 48000 take.wav`; stream 1 is APAC).
2. Find each sentence roughly with whisper (both `ggml-large-v3-turbo-q5_0.bin` and `ggml-small.en.bin`,
   `-ml 1 -sow -ojf`). The large model can merge a restart into one sentence; the small one keeps both
   reads, so compare the two. Whisper's word times drift, so use them only to locate a sentence.
3. Place every cut in a silence from the energy scan (`--silences FILE`), keep the last clean read of
   every sentence, and write one entry per kept stretch with a comment saying why the cut sits there.
4. Pick a pause of room noise only (no breath) for NOISE_SAMPLE.
5. Run this script, then `python3 video/scripts/timeline.py narration video/public/audio/narration-assembled.wav --clean`
   and read its "extra words heard" line.

Usage, from the repository root:
    python3 video/scripts/assemble_narration.py                    # build narration-assembled.wav from PIECES
    python3 video/scripts/assemble_narration.py --silences FILE     # list the silences of one take
    python3 video/scripts/assemble_narration.py --ab START          # 15 s before/after clips into video/out/
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
AB_DIR = ROOT / "video/out"
RATE = 48_000
TARGET_LUFS = -16.0
FADE_S = 0.03
SENTENCE_GAP_S = 0.55
PARAGRAPH_GAP_S = 0.8
LONG_PAUSE_S = 1.2
KEEP_PAUSE_S = 0.9
TEMPO = 1.025  # pitch kept; at 1.0 the cut is 176.8 s, 182.8 s with the 2 s lead-in and 4 s end hold, over 3:00

# Noise reduction. NR_DB is afftdn's reduction; the gate lowers what stays below GATE_DB by at most
# GATE_RANGE_DB, with a 5 ms attack so word onsets and consonants pass untouched.
NR_DB = 24
NF_DB = -40
GATE_DB = -44
GATE_RANGE_DB = 12

# A pause of room noise only, per take, for the denoiser to learn the noise from (seconds in the take).
NOISE_SAMPLE = {
    # 2.1 s between the abandoned first read of the FalkorDB paragraph and its restart; measured -37.6 dBFS RMS.
    "take-01.wav": (147.75, 149.85),
}

# (file in video/recordings/, start_s, end_s, gap_after_s, label). take-01 is one read of the whole
# script with three restarts; each restart keeps the later read. Cuts sit in silences from the energy
# scan (10 ms RMS, 50 ms profile), never on whisper word times.
PIECES: list[tuple[str, float, float, float, str]] = [
    # Paragraphs 1 to 7 up to "...they failed it five times." The take opens with 0.9 s of room noise
    # (first word at 0.94 s). "times" ends at 125.17 s, where a 0.7 s silence starts.
    ("take-01.wav", 0.85, 125.22, SENTENCE_GAP_S, "Day one on a new team ... and they failed it five times."),
    # 125.9 to 131.2 s is a first read of the next sentence ("... drew a fresh example"), then 2.6 s of
    # near-silence; the second read starts at 133.90 s (onset rises from -36 to -15 dBFS). Keep the second.
    ("take-01.wav", 133.80, 141.32, PARAGRAPH_GAP_S, "Each time I fixed the extraction or the audit rules ... on the record."),
    # 142.2 to 147.6 s is a first read of the FalkorDB paragraph that stops after "A read-only...";
    # 2.4 s of silence follows and the restart begins at 150.00 s. Keep the restart, through "its history."
    # which ends at 190.05 s.
    ("take-01.wav", 149.92, 190.08, SENTENCE_GAP_S, "FalkorDB does real work here ... you inherit its history."),
    # "inherit-brain lets you ask it." is read twice: 190.6 to 192.6 s straight after "history", then
    # again from 194.88 s after a 2.3 s pause. Keep the last read; the take ends in silence at 197.6 s.
    ("take-01.wav", 194.84, 197.65, 0.0, "inherit-brain lets you ask it."),
]


def _run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, **kw)


def _loudness(path: Path) -> dict:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "loudnorm=print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return json.loads(out[out.rindex("{"): out.rindex("}") + 1])


def rms_db(path: Path, start: float, end: float) -> float:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-ss", str(start), "-to", str(end), "-i", str(path), "-af", "astats", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return float(next(l for l in out.splitlines() if "RMS level dB" in l).split()[-1])


def denoise(source: Path, target: Path, sample: tuple[float, float]) -> None:
    """High-pass, afftdn with a noise profile learned from `sample` (prepended, then trimmed off), gentle gate."""
    s0, s1 = sample
    d = s1 - s0
    gate = f"agate=threshold={10 ** (GATE_DB / 20):.5f}:range={10 ** (-GATE_RANGE_DB / 20):.4f}:ratio=2:attack=5:release=250:knee=4"
    graph = (f"[0:a]asplit=2[a][b];[b]atrim={s0}:{s1},asetpts=PTS-STARTPTS[n];"
             f"[n][a]concat=n=2:v=0:a=1,highpass=f=80,"
             f"asendcmd=c='0.05 afftdn sn start;{d - 0.05:.2f} afftdn sn stop',afftdn=nr={NR_DB}:nf={NF_DB},"
             f"atrim=start={d},asetpts=PTS-STARTPTS,{gate}[out]")
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-filter_complex", graph, "-map", "[out]", "-ac", "1", "-ar", str(RATE), str(target)])


def _levels(source: Path):
    import numpy as np

    raw = _run(["ffmpeg", "-v", "error", "-i", str(source), "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-"], capture_output=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    hop = RATE // 100  # 10 ms
    frames = len(audio) // hop
    level = 20 * np.log10(np.sqrt((audio[:frames * hop].reshape(frames, hop) ** 2).mean(1)) + 1e-9)
    # Quiet = within 10 dB of the take's quietest tenth: robust to a raised noise floor.
    return audio, hop, frames, level < np.percentile(level, 10) + 10


def silences(source: Path, minimum_s: float = 0.25) -> list[tuple[float, float]]:
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


def _shorten_pauses(source: Path, target: Path) -> tuple[float, int]:
    """Shorten every pause longer than LONG_PAUSE_S to KEEP_PAUSE_S, cutting from its middle."""
    import numpy as np

    audio, hop, frames, quiet = _levels(source)
    keep = np.ones(len(audio), dtype=bool)
    removed, count, start = 0.0, 0, None
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
                count += 1
            start = None
    _run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-i", "-", str(target)], input=audio[keep].tobytes())
    return removed, count


def _normalise(source: Path, target: Path, extra: str = "") -> None:
    """Two-pass loudnorm to TARGET_LUFS, -1.5 dBTP."""
    m = _loudness(source)
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-af",
          (extra + "," if extra else "") +
          f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true",
          "-ar", str(RATE), "-ac", "1", str(target)])


def main() -> None:
    if not PIECES:
        raise SystemExit("PIECES is empty: fill it from the takes in video/recordings/ (see the instructions at the top of this file)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cleaned: dict[str, Path] = {}
        for name in dict.fromkeys(p[0] for p in PIECES):
            source = RECORDINGS / name
            cleaned[name] = tmpdir / f"clean-{len(cleaned)}.wav"
            denoise(source, cleaned[name], NOISE_SAMPLE[name])
            s0, s1 = NOISE_SAMPLE[name]
            print(f"{name}: noise floor {rms_db(source, s0, s1):.1f} dBFS before, {rms_db(cleaned[name], s0, s1):.1f} dBFS after"
                  f" (speech {_loudness(source)['input_i']} LUFS before, {_loudness(cleaned[name])['input_i']} LUFS after)")
        parts: list[Path] = []
        clock = 0.0
        for index, (name, start, end, gap, label) in enumerate(PIECES):
            length = end - start
            piece = tmpdir / f"{index:02d}.wav"
            _run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-to", str(end), "-i", str(cleaned[name]), "-af",
                  f"afade=t=in:d={FADE_S},afade=t=out:st={length - FADE_S:.3f}:d={FADE_S}" + (f",apad=pad_dur={gap}" if gap else ""),
                  "-ac", "1", "-ar", str(RATE), str(piece)])
            print(f"{clock:7.2f}s  {length:6.2f}s  {start:7.2f} to {end:7.2f} in {name}  {label}")
            clock += length + gap
            parts.append(piece)
        listing = tmpdir / "parts.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        joined = tmpdir / "joined.wav"
        _run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)])
        tightened = tmpdir / "tightened.wav"
        removed, count = _shorten_pauses(joined, tightened)
        print(f"pauses: {count} longer than {LONG_PAUSE_S}s shortened to {KEEP_PAUSE_S}s, {removed:.1f}s removed")
        if not 1.0 <= TEMPO <= 1.04:
            raise SystemExit("TEMPO must stay between 1.0 and 1.04")
        _normalise(tightened, OUT, f"atempo={TEMPO}" if TEMPO != 1.0 else "")
    total = float(_run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(OUT)],
                       capture_output=True, text=True).stdout)
    print(f"{OUT.relative_to(ROOT)}: {clock:.1f}s joined, {total:.1f}s at tempo {TEMPO}, {_loudness(OUT)['input_i']} LUFS"
          " (the video adds a 2 s lead-in and a 4 s end hold)")


def ab_clips(start: float, seconds: float = 15.0) -> None:
    """The same stretch of the first take before and after cleaning, both at TARGET_LUFS, so only the noise differs."""
    name = PIECES[0][0]
    source = RECORDINGS / name
    AB_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        clean = tmpdir / "clean.wav"
        denoise(source, clean, NOISE_SAMPLE[name])
        for label, path in (("before", source), ("after", clean)):
            clip = tmpdir / f"{label}.wav"
            _run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(seconds), "-i", str(path), "-ac", "1", "-ar", str(RATE), str(clip)])
            _normalise(clip, AB_DIR / f"denoise-{label}.wav")
            print(f"{(AB_DIR / f'denoise-{label}.wav').relative_to(ROOT)}: {start:.1f} to {start + seconds:.1f} s of {name}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--silences":
        for a, b in silences(Path(sys.argv[2])):
            print(f"{a:8.2f} {b:8.2f}  ({b - a:.2f}s)")
    elif len(sys.argv) == 3 and sys.argv[1] == "--ab":
        ab_clips(float(sys.argv[2]))
    else:
        main()
