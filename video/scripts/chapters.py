"""Write YouTube chapter timestamps, one per beat, from the aligned timeline.

The first chapter starts at 0:00; each later one starts where its paragraph is first spoken.

Usage, from the repository root:
    python3 video/scripts/chapters.py video/out/chapters.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIMELINE = ROOT / "video/src/content/timeline.json"
TITLES = {
    "cold_open": "The question you inherit",
    "intro": "inherit-brain, one graph per product",
    "walk_graph": "Walking the graph",
    "benchmark": "HERB and the pre-registration",
    "arms": "Eight ways to answer",
    "results": "Results",
    "audit": "Six audit rounds",
    "falkordb": "What FalkorDB does here",
    "check_it": "Check every answer",
    "close": "Links",
}


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "video/out/chapters.txt"
    beats = json.loads(TIMELINE.read_text())["beats"]
    lines = []
    for index, beat in enumerate(beats):
        start = 0 if index == 0 else int(beat["start_s"])
        lines.append(f"{start // 60}:{start % 60:02d} {TITLES[beat['id']]}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{target}: {len(lines)} chapters")


if __name__ == "__main__":
    main()
