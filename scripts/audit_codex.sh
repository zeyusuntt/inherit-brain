#!/usr/bin/env bash
# Codex auditor runs for the dev acceptance audit (Task 13, design rulings R5 and R8).
# One fresh `codex exec` per batch, read-only sandbox, started in an empty directory that holds only that batch file,
# so the auditor's input is the batch (rubric plus packets) and nothing else. A reply that is not valid JSON is retried
# once; a second failure leaves no verdict file, and collect counts those packets as missing (R5).
#
# usage: scripts/audit_codex.sh <round> <batches|third_batches> <codex|third> [parallel=3] [effort=high]
set -euo pipefail
round=$1; src=$2; who=$3; par=${4:-3}; effort=${5:-high}
repo=$(cd "$(dirname "$0")/.." && pwd)
rd="$repo/runs/audit/round-$round"
out="$rd/verdicts/$who"; logs="$rd/logs"; work=$(mktemp -d)
mkdir -p "$out" "$logs"
[ -d "$rd/$src" ] || { echo "no $rd/$src"; exit 2; }

cat > "$work/prompt.txt" <<'EOF'
Read the file batch.txt in the current directory. It holds audit instructions followed by audit packets. Follow the instructions exactly and judge each packet only from its own text. Reply with only the JSON the instructions ask for: no prose and no code fences.
EOF

cat > "$work/extract.py" <<'EOF'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else ""
starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
if not starts:
    sys.exit(1)
try:
    obj, _ = json.JSONDecoder().raw_decode(text[min(starts):])
except ValueError:
    sys.exit(1)
json.dump(obj, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
EOF

run_one() {
  name=$1
  [ -s "$out/$name.json" ] && return 0  # resume: done already
  d="$work/$name"; mkdir -p "$d"; cp "$rd/$src/$name.txt" "$d/batch.txt"
  for attempt in 1 2; do
    codex exec -s read-only --skip-git-repo-check -c model_reasoning_effort="$effort" -C "$d" -o "$d/last.txt" - \
      < "$work/prompt.txt" > "$logs/$who-$name-$attempt.log" 2>&1 || true
    if [ -s "$d/last.txt" ] && python3 "$work/extract.py" "$d/last.txt" "$out/$name.json"; then
      echo "ok $name (attempt $attempt)"; return 0
    fi
  done
  echo "FAILED $name"
}
export -f run_one; export out logs work rd src who effort
ls "$rd/$src"/*.txt | xargs -n1 basename | sed 's/\.txt$//' | xargs -P "$par" -I{} bash -c 'run_one {}'
rm -rf "$work"
