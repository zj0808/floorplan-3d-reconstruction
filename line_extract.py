"""Wall-line extraction from 2D floor-plan rasters.

The extraction stage takes a raster floor plan and returns the structural
linework as a list of axis-aligned segments. Downstream, ``reconstruct.py``
extrudes those segments into wall geometry.

Design notes
------------
Floor plans are hostile input:

* wall strokes are broken by text labels, dimension ticks and door arcs,
* the same physical wall is usually drawn as two parallel strokes (wall
  thickness), which Hough happily reports as two unrelated lines,
* almost everything structural is orthogonal.

The pipeline therefore does gap bridging *before* edge detection, snaps
segments to the dominant axes *after* detection, and collapses duplicate
segments last. That ordering matters -- snapping before dedupe is what makes
the dedupe tolerance meaningful.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

Segment = Tuple[int, int, int, int]


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    """Read an image through ``fromfile``/``imdecode``.

    ``cv2.imread`` hands the path to the C++ layer as a narrow (ANSI) string, so
    it fails on non-ASCII paths on Windows -- which is exactly where project
    folders tend to live. Reading the bytes in Python and decoding from memory
    sidesteps the whole encoding problem.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"cannot read image: {path}")

    buffer = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buffer, flags)
    if img is None:
        raise FileNotFoundError(f"cannot decode image: {path}")
    return img


def imwrite_unicode(path: str | Path, img: np.ndarray) -> Path:
    """Encode and write an image without going through the ANSI-only path API.

    ``np.ndarray.tofile`` is deliberately avoided: it ignores the requested
    container and happily writes a different encoding into a ``.png`` name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower() or ".png"
    ok, buffer = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"cannot encode image for: {path}")

    with open(str(path), "wb") as fh:
        fh.write(buffer.tobytes())
    return path


def load_grayscale(image_path: str | Path) -> np.ndarray:
    """Read an image as uint8 grayscale, raising if the file is unreadable."""
    return imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)


def binarize(img: np.ndarray, threshold: int = 200) -> np.ndarray:
    """Inverse-threshold, so wall strokes become positive (255) pixels.

    Floor plans are nearly always dark linework on a light background. If that
    assumption flips for some inputs, the mean intensity check below inverts the
    result instead of silently returning an empty mask.
    """
    if float(img.mean()) < 127.0:
        img = cv2.bitwise_not(img)

    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def bridge_gaps(binary: np.ndarray, kernel_size: int = 3, iterations: int = 2) -> np.ndarray:
    """Morphological closing to reconnect strokes interrupted by text/dim ticks.

    A small rectangular kernel is deliberate: a large one would fuse adjacent
    parallel walls into a solid blob before edge detection runs.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def detect_segments(
    edges: np.ndarray,
    threshold: int = 60,
    min_len: int = 40,
    max_gap: int = 8,
) -> List[Segment]:
    """Run probabilistic Hough and return raw segments as (x1, y1, x2, y2)."""
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )
    if lines is None:
        return []
    return [tuple(int(v) for v in line[0]) for line in lines]  # type: ignore[misc]


def snap_axis_aligned(segments: Sequence[Segment], tol: int = 3) -> List[Segment]:
    """Snap near-vertical / near-horizontal segments onto their dominant axis.

    A near-vertical stroke becomes ``x = mean(x1, x2)``; a near-horizontal one
    becomes ``y = mean(y1, y2)``. Diagonal segments are kept as-is so real
    outliers stay visible instead of being forced into the grid.
    """
    snapped: List[Segment] = []
    for x1, y1, x2, y2 in segments:
        if abs(x1 - x2) <= tol:
            x = round((x1 + x2) / 2.0)
            snapped.append((x, min(y1, y2), x, max(y1, y2)))
            continue
        if abs(y1 - y2) <= tol:
            y = round((y1 + y2) / 2.0)
            snapped.append((min(x1, x2), y, max(x1, x2), y))
            continue
        snapped.append((x1, y1, x2, y2))
    return snapped


def dedupe_segments(segments: Sequence[Segment], tol: int = 6) -> List[Segment]:
    """Collapse segments that describe the same wall within ``tol`` pixels.

    Comparison is pairwise against the already-accepted set, so the first
    (usually the longest, because Hough votes by length) segment wins and short
    fragments of the same wall are dropped. The set stays tiny -- a floor plan
    yields tens of segments, not thousands -- so the quadratic scan is fine.
    """
    kept: List[Segment] = []
    for seg in sorted(segments, key=_segment_length, reverse=True):
        if any(_segments_match(seg, other, tol) for other in kept):
            continue
        kept.append(seg)
    return kept


def _segment_length(seg: Segment) -> float:
    x1, y1, x2, y2 = seg
    return float(np.hypot(x2 - x1, y2 - y1))


def _segments_match(a: Segment, b: Segment, tol: int) -> bool:
    return all(abs(p - q) <= tol for p, q in zip(a, b))


def extract_walls(
    image_path: str | Path,
    *,
    threshold: int = 200,
    canny_lo: int = 50,
    canny_hi: int = 150,
    min_len: int = 40,
    max_gap: int = 8,
    snap_tol: int = 3,
    dedupe_tol: int = 6,
) -> List[Segment]:
    """Full extraction pipeline: raster floor plan in, wall segments out."""
    img = load_grayscale(image_path)
    binary = binarize(img, threshold=threshold)
    closed = bridge_gaps(binary)
    edges = cv2.Canny(closed, canny_lo, canny_hi)

    raw = detect_segments(edges, min_len=min_len, max_gap=max_gap)
    snapped = snap_axis_aligned(raw, tol=snap_tol)
    return dedupe_segments(snapped, tol=dedupe_tol)


def render_debug(
    image_path: str | Path,
    segments: Sequence[Segment],
    out_path: str | Path,
) -> Path:
    """Write an annotated copy of the input with the extracted segments drawn."""
    color = imread_unicode(image_path, cv2.IMREAD_COLOR)

    for x1, y1, x2, y2 in segments:
        cv2.line(color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return imwrite_unicode(out_path, color)


def summarize(segments: Sequence[Segment]) -> str:
    vertical = sum(1 for x1, _, x2, _ in segments if x1 == x2)
    horizontal = sum(1 for _, y1, _, y2 in segments if y1 == y2)
    diagonal = len(segments) - vertical - horizontal
    return (
        f"{len(segments)} segments "
        f"({vertical} vertical, {horizontal} horizontal, {diagonal} diagonal)"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="path to a floor-plan raster")
    parser.add_argument("--threshold", type=int, default=200)
    parser.add_argument("--min-len", type=int, default=40)
    parser.add_argument("--dedupe-tol", type=int, default=6)
    parser.add_argument("--debug", action="store_true", help="write an annotated PNG")
    parser.add_argument("--out", default="out", help="output directory for debug images")
    args = parser.parse_args(argv)

    segments = extract_walls(
        args.image,
        threshold=args.threshold,
        min_len=args.min_len,
        dedupe_tol=args.dedupe_tol,
    )
    print(summarize(segments))
    for seg in segments:
        print(seg)

    if args.debug:
        out = Path(args.out) / (Path(args.image).stem + "_segments.png")
        written = render_debug(args.image, segments, out)
        print(f"debug image: {written}")

    return 0 if segments else 1


if __name__ == "__main__":
    sys.exit(main())
