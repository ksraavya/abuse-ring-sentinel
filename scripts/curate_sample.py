import json
from pathlib import Path

records = []
with open('artifacts/verifier/world_c/verification_records.jsonl') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

blocks = [r for r in records if str(r.get('policy', {}).get('action', '')).lower() == 'block']
reviews = [r for r in records if str(r.get('policy', {}).get('action', '')).lower() == 'review']
allows = [r for r in records if str(r.get('policy', {}).get('action', '')).lower() == 'allow']

blocks.sort(key=lambda r: -len(r.get('evidence', []) or []))

manifest = json.loads(Path('data/generated/world_c/manifest.json').read_text())
ring_accounts = {a for ring in manifest.get('rings', []) for a in ring.get('account_ids', [])}
ring_records = [r for r in records if r.get('account_id') in ring_accounts]
ring_blocks = [r for r in ring_records if str(r.get('policy', {}).get('action', '')).lower() == 'block']
ring_reviews = [r for r in ring_records if str(r.get('policy', {}).get('action', '')).lower() == 'review']

sample = []
seen = set()

def add(rec):
    eid = rec.get('event_id')
    if eid not in seen:
        seen.add(eid)
        sample.append(rec)

for r in ring_blocks[:30]: add(r)
for r in ring_reviews[:20]: add(r)

non_ring_blocks = [r for r in blocks if r.get('account_id') not in ring_accounts]
for r in non_ring_blocks[:20]: add(r)
for r in reviews[:50]: add(r)
for r in allows[:10]: add(r)

print(f'Total sample: {len(sample)}')
actions = [str(r.get('policy', {}).get('action', 'review')).lower() for r in sample]
print(f'Blocks: {actions.count("block")}')
print(f'Reviews: {actions.count("review")}')
print(f'Allows: {actions.count("allow")}')
ring_in_sample = {r.get('account_id') for r in sample if r.get('account_id') in ring_accounts}
print(f'Ring accounts in sample: {len(ring_in_sample)}')

with open('dashboard/sample_records.jsonl', 'w') as f:
    for r in sample:
        f.write(json.dumps(r) + '\n')
print('Written to dashboard/sample_records.jsonl')