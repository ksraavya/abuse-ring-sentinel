from dashboard.data import extract_metrics, manifest_graph


def test_extract_metrics_nested():
    obj={"detector":{"metrics":{"precision":0.5,"recall":0.8,"pr_auc":0.7}},"economic":{"economic_cost":1234}}
    got=extract_metrics(obj)
    assert got["precision"]==0.5
    assert got["recall"]==0.8
    assert got["pr_auc"]==0.7
    assert got["cost"]==1234


def test_manifest_star_graph():
    nodes,edges=manifest_graph({"topology":"star","account_ids":["a","b","c"]})
    assert [n["id"] for n in nodes]==["a","b","c"]
    assert {(e["source"],e["target"]) for e in edges}=={("a","b"),("a","c")}
