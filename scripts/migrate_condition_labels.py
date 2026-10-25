"""One-off: rename condition labels anon/monitored to unwarned/warned in a results file (prompts were identical).

usage: uv run python scripts/migrate_condition_labels.py runs/paper.jsonl
Rows keep their original `code` commit, the code that produced them.
"""

import json
import sys
from pathlib import Path

RENAME = {"anon": "unwarned", "monitored": "warned"}

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
for row in rows:
    if row.get("condition") in RENAME:
        old = row["condition"]
        row["condition"] = RENAME[old]
        row["trial"] = row["trial"].replace(f"|{old}|", f"|{RENAME[old]}|")
        row["labels_migrated"] = "anon/monitored -> unwarned/warned"
path.write_text("".join(json.dumps(row) + "\n" for row in rows))
print(f"{path}: {sum('labels_migrated' in r for r in rows)} rows relabelled")
