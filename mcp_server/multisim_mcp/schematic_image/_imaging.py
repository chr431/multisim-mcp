"""Small image-processing primitives implemented with numpy alone.

The Multisim Automation API is 32-bit only, so every line of server-side code
must run on a 32-bit interpreter -- and SciPy publishes no 32-bit Windows wheel.
Rather than make image reconstruction an optional half-feature that only works
on 64-bit installs, the handful of operations this package actually needs are
implemented here directly on numpy arrays.

Only four operations are required, and each has a tight, allocation-conscious
implementation:

``label_components``
    Connected-component labelling by row runs with a union-find over run
    overlaps.  This is linear in the number of runs rather than in pixels, which
    keeps a 30-megapixel sheet fast without allocating a per-pixel graph.
``erode_cross``
    Diamond erosion by repeated 4-neighbour intersection.
``dilate_runs``
    Horizontal or vertical dilation, used to merge glyphs into words.
``component_boxes`` / ``component_centroids``
    Bounding boxes and centres, derived from per-label reductions.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np


def _as_bool(mask: Any) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype != bool:
        array = array.astype(bool)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {array.shape}")
    return array


def row_runs(mask: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (row, start, end) for maximal True runs, ordered row-major.

    ``end`` is exclusive.  Runs never touch the image border because the input is
    padded by one zero column before differencing.
    """
    array = _as_bool(mask)
    rows, cols = array.shape
    padded = np.zeros((rows, cols + 2), dtype=np.int8)
    padded[:, 1:-1] = array
    delta = np.diff(padded, axis=1)
    run_rows, starts = np.nonzero(delta == 1)
    _, ends = np.nonzero(delta == -1)
    return run_rows, starts, ends


def label_components(mask: Any, *, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Label connected components; return (labels, count).

    ``labels`` is an int32 array the same shape as ``mask`` where 0 is background
    and 1..count are component ids, numbered in row-major order of first
    appearance.
    """
    array = _as_bool(mask)
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    rows, starts, ends = row_runs(array)
    run_count = int(rows.size)
    labels = np.zeros(array.shape, dtype=np.int32)
    if run_count == 0:
        return labels, 0

    # --- union-find over runs
    parent = np.arange(run_count, dtype=np.int64)

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = int(parent[root])
        while parent[item] != root:
            parent[item], item = root, int(parent[item])
        return root

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    # --- group run indices by their row, so adjacent rows can be merge-joined
    row_values, row_start_index = np.unique(rows, return_index=True)
    boundaries = np.append(row_start_index, run_count)

    gap = 1 if connectivity == 8 else 0
    for position in range(len(row_values) - 1):
        top_row = int(row_values[position])
        # Only adjacent rows can touch.
        if int(row_values[position + 1]) != top_row + 1:
            continue
        top = range(int(boundaries[position]), int(boundaries[position + 1]))
        bottom = range(int(boundaries[position + 1]), int(boundaries[position + 2]))
        top_list = list(top)
        bottom_list = list(bottom)
        i = j = 0
        while i < len(top_list) and j < len(bottom_list):
            a, b = top_list[i], bottom_list[j]
            # Overlap test with the connectivity gap folded in.
            if starts[a] < ends[b] + gap and starts[b] < ends[a] + gap:
                union(a, b)
                # A run on one row may touch several on the other; advance the
                # one that ends first.
                if ends[a] <= ends[b]:
                    i += 1
                else:
                    j += 1
            elif ends[a] <= starts[b]:
                i += 1
            else:
                j += 1

    # --- assign compact ids in order of first appearance
    roots = np.array([find(index) for index in range(run_count)], dtype=np.int64)
    _, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
    component_of_run = (inverse + 1).astype(np.int32)

    for index in range(run_count):
        labels[int(rows[index]), int(starts[index]) : int(ends[index])] = component_of_run[index]
    return labels, int(counts.size)


def erode_cross(mask: Any, iterations: int = 1) -> np.ndarray:
    """Erode by a 4-neighbour cross, ``iterations`` times.

    Equivalent to eroding by a diamond of radius ``iterations``, which is the
    shape needed to find strokes thicker than their neighbours.  Pixels outside
    the image count as background, so the outermost rows and columns are cleared
    on every iteration.
    """
    array = _as_bool(mask)
    result = array
    for _ in range(max(0, int(iterations))):
        out = result.copy()
        out[1:, :] &= result[:-1, :]
        out[:-1, :] &= result[1:, :]
        out[:, 1:] &= result[:, :-1]
        out[:, :-1] &= result[:, 1:]
        # The neighbour beyond each border is background, so the border itself
        # cannot survive an erosion.
        out[0, :] = False
        out[-1, :] = False
        out[:, 0] = False
        out[:, -1] = False
        result = out
    return result


def dilate_runs(mask: Any, *, horizontal: int = 0, vertical: int = 0) -> np.ndarray:
    """Dilate by a horizontal and/or vertical box of the given radii."""
    array = _as_bool(mask)
    result = array
    if horizontal > 0:
        out = result.copy()
        for shift in range(1, int(horizontal) + 1):
            out[:, shift:] |= result[:, :-shift]
            out[:, :-shift] |= result[:, shift:]
        result = out
    if vertical > 0:
        out = result.copy()
        for shift in range(1, int(vertical) + 1):
            out[shift:, :] |= result[:-shift, :]
            out[:-shift, :] |= result[shift:, :]
        result = out
    return result


def component_boxes(labels: Any, count: int) -> list[tuple[int, int, int, int] | None]:
    """Return ``(x0, y0, x1, y1)`` bounding boxes (x1/y1 exclusive) per label."""
    array = np.asarray(labels)
    if count <= 0:
        return []
    height, width = array.shape
    # Per-label extremes via segment-wise reductions.
    flat = array.reshape(-1)
    index = np.arange(flat.size)
    order = np.argsort(flat, kind="stable")
    sorted_labels = flat[order]
    sorted_index = index[order]
    boundaries = np.searchsorted(sorted_labels, np.arange(1, count + 2))

    boxes: list[tuple[int, int, int, int] | None] = []
    for label in range(1, count + 1):
        lo, hi = int(boundaries[label - 1]), int(boundaries[label])
        if lo >= hi:
            boxes.append(None)
            continue
        positions = sorted_index[lo:hi]
        ys, xs = np.divmod(positions, width)
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return boxes


def component_centroids(labels: Any, count: int) -> list[tuple[float, float]]:
    """Return ``(x, y)`` centres of mass per label."""
    array = np.asarray(labels)
    if count <= 0:
        return []
    flat = array.reshape(-1)
    width = array.shape[1]
    sums = np.bincount(flat, minlength=count + 1)
    index = np.arange(flat.size, dtype=np.float64)
    ys, xs = np.divmod(index, width)
    sum_x = np.bincount(flat, weights=xs, minlength=count + 1)
    sum_y = np.bincount(flat, weights=ys, minlength=count + 1)
    out: list[tuple[float, float]] = []
    for label in range(1, count + 1):
        total = sums[label]
        if total == 0:
            out.append((0.0, 0.0))
        else:
            out.append((float(sum_x[label] / total), float(sum_y[label] / total)))
    return out


def component_areas(labels: Any, count: int) -> np.ndarray:
    """Return the pixel area of each label, index 1..count."""
    array = np.asarray(labels)
    if count <= 0:
        return np.zeros(1, dtype=np.int64)
    return np.bincount(array.reshape(-1), minlength=count + 1)[: count + 1].astype(np.int64)


def count_holes(sub_mask: Any, *, connectivity: int = 8) -> int:
    """Count enclosed background regions inside ``sub_mask``.

    A hole is a background component that does not touch the sub-image border.
    """
    array = _as_bool(sub_mask)
    if array.size == 0:
        return 0
    padded = np.zeros((array.shape[0] + 2, array.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = array
    inverse = ~padded
    labels, count = label_components(inverse, connectivity=connectivity)
    if count == 0:
        return 0
    border = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    return int(count - border.size + (1 if 0 in border else 0))


def unique_labels_in(mask: Any, labels: Any) -> np.ndarray:
    """Return the label ids that occur where ``mask`` is True."""
    values = np.unique(np.asarray(labels)[_as_bool(mask)])
    return values[values > 0]


__all__ = [
    "component_areas",
    "component_boxes",
    "component_centroids",
    "count_holes",
    "dilate_runs",
    "erode_cross",
    "label_components",
    "row_runs",
    "unique_labels_in",
]
