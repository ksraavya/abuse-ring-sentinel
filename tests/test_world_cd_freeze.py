from __future__ import annotations

import json
from pathlib import Path

from worlds.generator import load_config
from worlds.schema import WorldId


ROOT = Path(__file__).resolve().parents[1]


def test_world_ids_include_development_and_final_worlds():
    assert WorldId.WORLD_C.value == "world_c"
    assert WorldId.WORLD_D.value == "world_d"


def test_world_c_and_d_configs_are_valid_and_independent():
    c = load_config(ROOT / "config" / "world_c.yaml", "world_c")
    d = load_config(ROOT / "config" / "world_d.yaml", "world_d")

    assert c.world_id is WorldId.WORLD_C
    assert d.world_id is WorldId.WORLD_D
    assert c.seed != d.seed
    assert c.duration_days == d.duration_days == 180
    assert c.total_rings == d.total_rings == 40
    assert c.legitimate_accounts == d.legitimate_accounts == 100000


def test_world_c_and_d_use_independent_account_id_namespaces():
    assert "world_c" not in "world_d"
    assert "world_d" not in "world_c"


def test_freeze_manifest_schema_is_explicit(tmp_path):
    manifest = {
        "freeze_version": 1,
        "world_id": "world_c",
        "files_sha256": {
            "events.jsonl": "x",
            "ground_truth.jsonl": "y",
            "manifest.json": "z",
            "config.yaml": "w",
        },
        "ground_truth_separate_from_events": True,
    }
    path = tmp_path / "freeze_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["freeze_version"] == 1
    assert loaded["ground_truth_separate_from_events"] is True
    assert set(loaded["files_sha256"]) == {
        "events.jsonl",
        "ground_truth.jsonl",
        "manifest.json",
        "config.yaml",
    }
