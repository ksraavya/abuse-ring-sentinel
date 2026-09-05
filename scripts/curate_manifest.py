# scripts/curate_manifest.py
import json
from pathlib import Path

manifest = json.loads(Path('data/generated/world_c/manifest.json').read_text())

slim = {
    "rings": [
        {
            "ring_id": r["ring_id"],
            "kind": r["kind"],
            "topology": r["topology"],
            "strength": r.get("strength", "moderate"),
            "account_ids": r["account_ids"],
            "activation_day": r.get("activation_day", 0),
            "coordination_start_day": r.get("coordination_start_day", 0),
            "coordination_end_day": r.get("coordination_end_day", 0),
        }
        for r in manifest.get("rings", [])
    ]
}

Path('dashboard/ring_manifest.json').write_text(json.dumps(slim, indent=2))
print(f'Rings: {len(slim["rings"])}')
total_accounts = sum(len(r["account_ids"]) for r in slim["rings"])
print(f'Total ring accounts: {total_accounts}')
size = len(json.dumps(slim)) / 1024
print(f'File size: {size:.1f} KB')