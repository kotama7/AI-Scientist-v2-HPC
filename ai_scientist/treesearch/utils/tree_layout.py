"""Tree layout and graph utilities for visualization."""

from __future__ import annotations

import numpy as np
from igraph import Graph

from ..journal import Journal


def get_edges(journal: Journal, include_none_root: bool = False):
    """Generate edges from parent-child relationships.

    If include_none_root is True, adds a virtual "None" root node at index len(journal)
    and connects all parentless nodes to it, including nodes inherited from a previous
    stage's best node. This ensures no nodes are isolated in the tree visualization.
    """
    for node in journal:
        for c in node.children:
            yield (node.step, c.step)

    if include_none_root:
        # Add edges from None root (at index len(journal)) to all parentless nodes,
        # including inherited nodes so they are not isolated in the tree
        none_root_idx = len(journal)
        for node in journal:
            if node.parent is None:
                yield (none_root_idx, node.step)


def generate_layout(n_nodes: int, edges: list, layout_type: str = "rt") -> np.ndarray:
    """Generate visual layout of graph using igraph."""
    layout = Graph(
        n_nodes,
        edges=edges,
        directed=True,
    ).layout(layout_type)
    y_max = max(layout[k][1] for k in range(n_nodes))
    layout_coords = []
    for n in range(n_nodes):
        layout_coords.append((layout[n][0], 2 * y_max - layout[n][1]))
    return np.array(layout_coords)


def normalize_layout(layout: np.ndarray) -> np.ndarray:
    """Normalize layout coordinates to [0, 1] range."""
    layout = (layout - layout.min(axis=0)) / (layout.max(axis=0) - layout.min(axis=0))
    layout[:, 1] = 1 - layout[:, 1]
    layout[:, 1] = np.nan_to_num(layout[:, 1], nan=0)
    layout[:, 0] = np.nan_to_num(layout[:, 0], nan=0.5)
    return layout
