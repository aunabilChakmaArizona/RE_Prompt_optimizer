import json
from pathlib import Path
from typing import Dict, List, Optional

POPULATION_PATH = Path(
    "trainings/20260128_151114_Qwen-Qwen3-4B/population.json"
)
OUTPUT_DIR = Path("tree_json")


def format_metric_pair(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None or std is None:
        return ""
    return f"{mean * 100:.2f}±{std * 100:.2f}"


def format_metrics(metrics: Optional[Dict[str, float]]) -> Dict[str, str]:
    if not metrics:
        return {
            "precision": "",
            "recall": "",
            "f1": "",
        }
    return {
        "precision": format_metric_pair(
            metrics.get("precision_mean"), metrics.get("precision_std")
        ),
        "recall": format_metric_pair(
            metrics.get("recall_mean"), metrics.get("recall_std")
        ),
        "f1": format_metric_pair(
            metrics.get("f1_mean"), metrics.get("f1_std")
        ),
    }


def build_tree(population: List[Dict[str, object]]) -> List[Dict[str, object]]:
    nodes: Dict[int, Dict[str, object]] = {}
    children_map: Dict[int, List[int]] = {}
    roots: List[int] = []

    for node in population:
        node_id = node.get("node_id")
        if node_id is None:
            continue
        nodes[int(node_id)] = node
        children_ids = node.get("children_ids") or []
        children_map[int(node_id)] = [int(child_id) for child_id in children_ids]

    for node in population:
        node_id = node.get("node_id")
        if node_id is None:
            continue
        parent_id = node.get("parent_id")
        if parent_id is None or int(parent_id) not in nodes:
            roots.append(int(node_id))

    def build_node(node_id: int) -> Dict[str, object]:
        node = nodes[node_id]
        val_score = node.get("val_score")
        metrics = format_metrics(val_score)
        return {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "children": [build_node(child_id) for child_id in children_map.get(node_id, [])],
        }

    return [build_node(node_id) for node_id in roots]


def load_population(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("population", [])


def main() -> int:
    population = load_population(POPULATION_PATH)
    tree = build_tree(population)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{POPULATION_PATH.stem}_tree.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(tree, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
