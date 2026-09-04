from __future__ import annotations

from streaming.evaluation import EVALUATION_PARTITIONS, EVALUATION_WORLD


def test_world_b_evaluation_contract_is_single_partition() -> None:
    assert EVALUATION_WORLD == "world_b"
    assert EVALUATION_PARTITIONS == 1
