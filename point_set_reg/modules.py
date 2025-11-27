import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


def ensure_binary(page: np.ndarray, thresh: int | None = None) -> np.ndarray:
    if page.ndim == 3:
        page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    if thresh is None:
        _, binary = cv2.threshold(page, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(page, thresh, 255, cv2.THRESH_BINARY)
    return binary


def foreground_mask(page: np.ndarray) -> np.ndarray:
    binary = ensure_binary(page)
    return binary < 128


def close_gaps(mask: np.ndarray, orientation: str, gap: int) -> np.ndarray:
    if orientation == "horizontal":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap))
    closed = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    return closed > 0


def runs_1d(values: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(values):
        if v:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - 1))
                start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def group_segments_by_coord(
    segments: List[Tuple[int, int, int]],
    merge_tol: int,
    orientation: str,
) -> List[Dict]:
    if not segments:
        return []

    segments_sorted = sorted(segments)
    groups: List[List[Tuple[int, int, int]]] = []
    current = [segments_sorted[0]]

    for seg in segments_sorted[1:]:
        coord, s, e = seg
        last_coord, _, _ = current[-1]
        if abs(coord - last_coord) <= merge_tol:
            current.append(seg)
        else:
            groups.append(current)
            current = [seg]
    groups.append(current)

    lines: List[Dict] = []
    for g in groups:
        coords = [c for c, _, _ in g]
        starts = [s for _, s, _ in g]
        ends = [e for _, _, e in g]

        seg_min = min(starts)
        seg_max = max(ends)

        if orientation == "horizontal":
            y_min = min(coords)
            y_max = max(coords)
            lines.append(
                {
                    "orientation": "horizontal",
                    "segments": g,
                    "y_min": y_min,
                    "y_max": y_max,
                    "x0": seg_min,
                    "x1": seg_max,
                    "length": seg_max - seg_min + 1,
                }
            )
        else:
            x_min = min(coords)
            x_max = max(coords)
            lines.append(
                {
                    "orientation": "vertical",
                    "segments": g,
                    "x_min": x_min,
                    "x_max": x_max,
                    "y0": seg_min,
                    "y1": seg_max,
                    "length": seg_max - seg_min + 1,
                }
            )

    return lines


def detect_horizontal_lines(
    page: np.ndarray,
    gap: int = 5,
    min_run_ratio: float = 0.15,
    merge_tol: int = 3,
) -> Tuple[List[Dict], np.ndarray]:
    mask = foreground_mask(page)
    closed = close_gaps(mask, "horizontal", gap)

    h, w = closed.shape
    min_run = int(w * min_run_ratio)

    segments: List[Tuple[int, int, int]] = []
    for y in range(h):
        row = closed[y]
        for s, e in runs_1d(row):
            if e - s + 1 >= min_run:
                segments.append((y, s, e))

    lines = group_segments_by_coord(segments, merge_tol, "horizontal")
    return lines, closed


def detect_vertical_lines(
    page: np.ndarray,
    gap: int = 5,
    min_run_ratio: float = 0.15,
    merge_tol: int = 3,
) -> Tuple[List[Dict], np.ndarray]:
    mask = foreground_mask(page)
    closed = close_gaps(mask, "vertical", gap)

    h, w = closed.shape
    min_run = int(h * min_run_ratio)

    segments: List[Tuple[int, int, int]] = []
    for x in range(w):
        col = closed[:, x]
        for s, e in runs_1d(col):
            if e - s + 1 >= min_run:
                segments.append((x, s, e))

    lines = group_segments_by_coord(segments, merge_tol, "vertical")
    return lines, closed


def sample_horizontal_line(line: Dict, mask: np.ndarray, num_samples: int) -> np.ndarray:
    x0 = line["x0"]
    x1 = line["x1"]
    y_min = line["y_min"]
    y_max = line["y_max"]

    xs = np.linspace(x0, x1, num_samples)
    ys = []

    for x in xs:
        xi = int(round(x))
        xi = max(0, min(mask.shape[1] - 1, xi))

        col = mask[y_min : y_max + 1, xi]
        ys_idx = np.where(col)[0]
        if len(ys_idx) == 0:
            ys.append(0.5 * (y_min + y_max))
        else:
            ys.append(y_min + np.median(ys_idx))

    xs = xs.astype(float)
    ys = np.array(ys, dtype=float)
    pts = np.stack([xs, ys], axis=1)
    return pts.astype(np.float32)


def sample_vertical_line(line: Dict, mask: np.ndarray, num_samples: int) -> np.ndarray:
    y0 = line["y0"]
    y1 = line["y1"]
    x_min = line["x_min"]
    x_max = line["x_max"]

    ys = np.linspace(y0, y1, num_samples)
    xs = []

    for y in ys:
        yi = int(round(y))
        yi = max(0, min(mask.shape[0] - 1, yi))

        row = mask[yi, x_min : x_max + 1]
        xs_idx = np.where(row)[0]
        if len(xs_idx) == 0:
            xs.append(0.5 * (x_min + x_max))
        else:
            xs.append(x_min + np.median(xs_idx))

    ys = ys.astype(float)
    xs = np.array(xs, dtype=float)
    pts = np.stack([xs, ys], axis=1)
    return pts.astype(np.float32)


def sample_points_on_lines(
    lines_h: List[Dict],
    mask_h: np.ndarray,
    lines_v: List[Dict],
    mask_v: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    pts_list: List[np.ndarray] = []

    for line in lines_h:
        pts_list.append(sample_horizontal_line(line, mask_h, num_samples))

    for line in lines_v:
        pts_list.append(sample_vertical_line(line, mask_v, num_samples))

    if not pts_list:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.concatenate(pts_list, axis=0)
    return pts.astype(np.float32)

def filter_points_by_margin(points: np.ndarray,
                            img_shape: tuple[int, int],
                            border_margin: int) -> np.ndarray:
    if border_margin <= 0 or points.size == 0:
        return points
    h, w = img_shape
    m = border_margin
    xs = points[:, 0]
    ys = points[:, 1]
    mask = (
        (xs >= m) &
        (xs <= w - 1 - m) &
        (ys >= m) &
        (ys <= h - 1 - m)
    )
    return points[mask]

def line_points(
    page: np.ndarray,
    num_samples_per_line: int = 10,
    detect_vertical: bool = True,
    detect_horizontal: bool = True,
    gap: int = 5,
    min_run_ratio_h: float = 0.15,
    min_run_ratio_v: float = 0.2,
    merge_tol: int = 3,
    border_margin: int = 0,
) -> np.ndarray:
    lines_h: List[Dict] = []
    lines_v: List[Dict] = []
    binary = ensure_binary(page)
    mask_h = np.zeros_like(binary, dtype=bool)
    mask_v = np.zeros_like(binary, dtype=bool)

    if detect_horizontal:
        lines_h, mask_h = detect_horizontal_lines(
            page,
            gap=gap,
            min_run_ratio=min_run_ratio_h,
            merge_tol=merge_tol,
        )

    if detect_vertical:
        lines_v, mask_v = detect_vertical_lines(
            page,
            gap=gap,
            min_run_ratio=min_run_ratio_v,
            merge_tol=merge_tol,
        )

    pts = sample_points_on_lines(
        lines_h,
        mask_h,
        lines_v,
        mask_v,
        num_samples_per_line,
    )

    pts = filter_points_by_margin(pts, binary.shape, border_margin)
    return pts