"""Generate the synthetic sample floor plan used by the demo.

Kept as a script rather than a checked-in binary so the sample is reproducible
and obviously synthetic -- no third-party plan images are redistributed here.

The drawing deliberately contains the noise the extractor has to survive:
thin dimension ticks outside the plan, two text labels straddling wall lines,
a door arc, and a furniture rectangle inside a room.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 900, 700
WALL = 3          # wall stroke thickness in px
THIN = 1
INK = 0           # dark linework on a light background
PAPER = 255       # background


def draw_plan() -> np.ndarray:
    img = np.full((HEIGHT, WIDTH), PAPER, dtype=np.uint8)

    # Outer shell: 80,80 -> 820,620
    cv2.rectangle(img, (80, 80), (820, 620), INK, WALL)

    # Interior partitions
    cv2.line(img, (420, 80), (420, 360), INK, WALL)     # vertical divider
    cv2.line(img, (420, 360), (820, 360), INK, WALL)    # horizontal divider
    cv2.line(img, (80, 460), (520, 460), INK, WALL)     # bottom-left room

    # Door gap: erase part of the bottom-left wall, then draw the arc
    cv2.line(img, (600, 460), (680, 460), PAPER, WALL + 2)
    cv2.ellipse(img, (600, 460), (80, 80), 0, 0, 90, INK, THIN)

    # Window marker on the top wall
    cv2.line(img, (250, 74), (250, 86), INK, THIN)
    cv2.line(img, (400, 74), (400, 86), INK, THIN)
    cv2.line(img, (250, 80), (400, 80), INK, THIN)

    # Furniture rectangle inside the top-right room
    cv2.rectangle(img, (560, 150), (760, 280), INK, THIN)

    # Text labels straddling walls -- this is what the gap-bridging stage fixes
    cv2.putText(img, "BEDROOM", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, INK, 2, cv2.LINE_AA)
    cv2.putText(img, "LIVING", (560, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.9, INK, 2, cv2.LINE_AA)

    # Dimension ticks outside the shell
    for x in range(100, 821, 100):
        cv2.line(img, (x, 640), (x, 660), INK, THIN)
    cv2.line(img, (80, 650), (820, 650), INK, THIN)
    cv2.putText(img, "9.2 m", (400, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.6, INK, 1, cv2.LINE_AA)

    return img


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "plan_demo.png"

    ok, buffer = cv2.imencode(".png", draw_plan())
    if not ok:
        raise IOError(f"cannot encode sample image for: {out_path}")

    # Write the encoded bytes directly; the path may contain non-ASCII characters.
    with open(str(out_path), "wb") as fh:
        fh.write(buffer.tobytes())

    print(f"wrote {out_path} ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
