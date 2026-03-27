from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


def expression_embedding(matrix, n_components: int = 64, seed: int = 1729) -> np.ndarray:
    n_components = min(n_components, max(2, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    return svd.fit_transform(matrix)


def knn_edges(features: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(features)), metric="cosine")
    nn.fit(features)
    indices = nn.kneighbors(return_distance=False)
    sources, targets = [], []
    for source, neighbors in enumerate(indices):
        for target in neighbors[1:]:
            sources.append(source)
            targets.append(int(target))
    return np.vstack([sources, targets]).astype(np.int64)


def spatial_edges(xy: np.ndarray, k: int) -> np.ndarray:
    if len(xy) < 2:
        return np.zeros((2, 0), dtype=np.int64)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(xy)), metric="euclidean")
    nn.fit(xy)
    indices = nn.kneighbors(return_distance=False)
    sources, targets = [], []
    for source, neighbors in enumerate(indices):
        for target in neighbors[1:]:
            sources.append(source)
            targets.append(int(target))
    return np.vstack([sources, targets]).astype(np.int64)


def build_graph_payload(
    node_features: np.ndarray,
    node_metadata: pd.DataFrame,
    expression_edge_index: np.ndarray | None = None,
    spatial_edge_index: np.ndarray | None = None,
) -> dict[str, Any]:
    edge_blocks = []
    edge_types = []
    if expression_edge_index is not None and expression_edge_index.size:
        edge_blocks.append(expression_edge_index)
        edge_types.extend(["expression"] * expression_edge_index.shape[1])
    if spatial_edge_index is not None and spatial_edge_index.size:
        edge_blocks.append(spatial_edge_index)
        edge_types.extend(["spatial"] * spatial_edge_index.shape[1])
    merged_edges = np.concatenate(edge_blocks, axis=1) if edge_blocks else np.zeros((2, 0), dtype=np.int64)
    return {
        "x": torch.tensor(node_features, dtype=torch.float32),
        "edge_index": torch.tensor(merged_edges, dtype=torch.long),
        "edge_type": edge_types,
        "node_metadata": node_metadata.reset_index(drop=True).to_dict(orient="list"),
    }


def save_graph_payload(payload: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return out


def edge_table(edge_index: np.ndarray, edge_type: str) -> pd.DataFrame:
    if edge_index.size == 0:
        return pd.DataFrame(columns=["source", "target", "edge_type"])
    return pd.DataFrame(
        {
            "source": edge_index[0].astype(int),
            "target": edge_index[1].astype(int),
            "edge_type": edge_type,
        }
    )


def feature_table(node_ids: pd.Series | list[str], features: np.ndarray, prefix: str) -> pd.DataFrame:
    node_ids = pd.Series(node_ids, name="node_id").astype(str)
    table = pd.DataFrame(features, columns=[f"{prefix}_{idx:03d}" for idx in range(features.shape[1])])
    table.insert(0, "node_id", node_ids.to_numpy())
    return table


def graph_summary(payload: dict[str, Any], donor_id: str, region: str, modality: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "donor_id": donor_id,
                "region": region,
                "modality": modality,
                "n_nodes": int(payload["x"].shape[0]),
                "n_edges": int(payload["edge_index"].shape[1]),
                "n_features": int(payload["x"].shape[1]),
            }
        ]
    )
