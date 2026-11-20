"""Wall graph -> 3D mesh extrusion.

Consumes the axis-aligned segments produced by ``line_extract`` and extrudes
them into vertical wall quads.

Status
------
The extrusion itself is implemented and runnable end to end. What is still open
(the part being built out with Devin Bot) is the *topology* stage: recognising
which segments are the two faces of one physical wall, joining segments at
corners, and dropping the inner face where walls meet. Right now each extracted
segment extrudes independently, which produces a correct-looking but slightly
redundant shell.

That split is deliberate -- the contract (segment in, mesh out) is stable, so
the topology work can land without touching the writer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

from line_extract import Segment, extract_walls

Vec3 = Tuple[float, float, float]


@dataclass
class WallMesh:
    """Accumulated geometry: vertices, quad faces, and the source wall graph."""

    vertices: List[Vec3] = field(default_factory=list)
    faces: List[Tuple[int, int, int, int]] = field(default_factory=list)
    walls: List[dict] = field(default_factory=list)

    def add_wall(self, segment: Segment, height: float, thickness: float = 0.0) -> None:
        """Extrude one segment into a vertical quad (optionally thick-walled).

        Image coordinates are converted to a right-handed Y-up frame: image
        ``x`` maps to world ``x``, image ``y`` maps to world ``-z`` so that the
        plan reads correctly when viewed from above, and the extrusion runs
        along ``+y``.
        """
        x1, y1, x2, y2 = segment
        base = len(self.vertices)

        if thickness <= 0.0:
            # Zero-thickness single face: two bottom verts, two top verts.
            quad = [
                (float(x1), 0.0, float(-y1)),
                (float(x2), 0.0, float(-y2)),
                (float(x2), height, float(-y2)),
                (float(x1), height, float(-y1)),
            ]
            self.vertices.extend(quad)
            self.faces.append((base, base + 1, base + 2, base + 3))
        else:
            # Offset perpendicular to the segment on both sides to get a box.
            dx, dy = float(x2 - x1), float(y2 - y1)
            length = math.hypot(dx, dy)
            if length == 0.0:
                return
            nx, ny = -dy / length * thickness / 2.0, dx / length * thickness / 2.0

            corners = [
                (x1 + nx, y1 + ny),
                (x2 + nx, y2 + ny),
                (x2 - nx, y2 - ny),
                (x1 - nx, y1 - ny),
            ]
            for cx, cy in corners:
                self.vertices.append((float(cx), 0.0, float(-cy)))
            for cx, cy in corners:
                self.vertices.append((float(cx), height, float(-cy)))

            # Four side faces plus top and bottom caps.
            self.faces.extend(
                [
                    (base + 0, base + 1, base + 5, base + 4),
                    (base + 1, base + 2, base + 6, base + 5),
                    (base + 2, base + 3, base + 7, base + 6),
                    (base + 3, base + 0, base + 4, base + 7),
                    (base + 4, base + 5, base + 6, base + 7),
                    (base + 0, base + 3, base + 2, base + 1),
                ]
            )

        self.walls.append(
            {
                "segment": list(segment),
                "length": round(math.hypot(x2 - x1, y2 - y1), 3),
                "height": height,
                "thickness": thickness,
            }
        )

    def to_obj(self, comment: str = "floorplan-3d-reconstruction") -> str:
        lines = [f"# {comment}"]
        for x, y, z in self.vertices:
            lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
        lines.append("")
        for a, b, c, d in self.faces:
            # OBJ is 1-indexed.
            lines.append(f"f {a + 1} {b + 1} {c + 1} {d + 1}")
        return "\n".join(lines) + "\n"

    def to_graph(self) -> dict:
        return {
            "vertex_count": len(self.vertices),
            "face_count": len(self.faces),
            "wall_count": len(self.walls),
            "walls": self.walls,
        }


def build_mesh(segments: Sequence[Segment], height: float = 2.8, thickness: float = 0.0) -> WallMesh:
    mesh = WallMesh()
    for segment in segments:
        mesh.add_wall(segment, height=height, thickness=thickness)
    return mesh


def write_outputs(mesh: WallMesh, out_dir: str | Path, stem: str) -> Tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_path = out_dir / f"{stem}.obj"
    json_path = out_dir / f"{stem}_graph.json"

    obj_path.write_text(mesh.to_obj(stem), encoding="utf-8")
    json_path.write_text(json.dumps(mesh.to_graph(), indent=2), encoding="utf-8")
    return obj_path, json_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="path to a floor-plan raster")
    parser.add_argument("--height", type=float, default=2.8, help="wall height in metres")
    parser.add_argument("--thickness", type=float, default=0.0, help="wall thickness, 0 = single face")
    parser.add_argument("--out", default="out", help="output directory")
    args = parser.parse_args(argv)

    segments = extract_walls(args.image)
    if not segments:
        print("no segments extracted -- check the input image", file=sys.stderr)
        return 1

    mesh = build_mesh(segments, height=args.height, thickness=args.thickness)
    stem = Path(args.image).stem
    obj_path, json_path = write_outputs(mesh, args.out, stem)

    print(f"walls: {len(mesh.walls)}  vertices: {len(mesh.vertices)}  faces: {len(mesh.faces)}")
    print(f"mesh:  {obj_path}")
    print(f"graph: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
