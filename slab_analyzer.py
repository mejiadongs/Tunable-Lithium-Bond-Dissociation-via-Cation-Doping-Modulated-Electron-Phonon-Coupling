from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CSV_COLUMNS = (
    "folder",
    "poscar_path",
    "total_atoms",
    "chemical_formula",
    "cell_volume",
    "surface_area",
    "vacuum_thickness",
)


@dataclass
class PoscarStructure:
    path: Path
    elements: list[str]
    counts: list[int]
    lattice: list[list[float]]
    fractional_coordinates: list[list[float]]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(dot(vector, vector))


def determinant_3x3(matrix: Sequence[Sequence[float]]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def inverse_3x3(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    determinant = determinant_3x3(matrix)
    if math.isclose(determinant, 0.0, abs_tol=1e-12):
        raise ValueError("Lattice matrix is singular and cannot be inverted.")

    return [
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) / determinant,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) / determinant,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) / determinant,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) / determinant,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) / determinant,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) / determinant,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) / determinant,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) / determinant,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) / determinant,
        ],
    ]


def multiply_row_vector_matrix(vector: Sequence[float], matrix: Sequence[Sequence[float]]) -> list[float]:
    return [sum(vector[row] * matrix[row][column] for row in range(3)) for column in range(3)]


def apply_cartesian_scale(vector: Sequence[float], axis_scales: Sequence[float]) -> list[float]:
    return [value * scale for value, scale in zip(vector, axis_scales)]


def parse_scaling(line: str, raw_lattice: Sequence[Sequence[float]]) -> tuple[list[float], list[list[float]]]:
    values = [float(token) for token in line.split()]
    if not values:
        raise ValueError("POSCAR scaling line is empty.")

    if len(values) == 1:
        scale = values[0]
        if math.isclose(scale, 0.0, abs_tol=1e-12):
            raise ValueError("POSCAR scaling factor cannot be zero.")
        if scale < 0:
            raw_volume = abs(determinant_3x3(raw_lattice))
            if math.isclose(raw_volume, 0.0, abs_tol=1e-12):
                raise ValueError("Raw lattice volume must be non-zero for negative scaling factors.")
            scale = (abs(scale) / raw_volume) ** (1.0 / 3.0)
        axis_scales = [scale, scale, scale]
    elif len(values) == 3:
        axis_scales = values
    else:
        raise ValueError("POSCAR scaling line must contain one or three numeric values.")

    scaled_lattice = [apply_cartesian_scale(vector, axis_scales) for vector in raw_lattice]
    return axis_scales, scaled_lattice


def parse_vector(line: str) -> list[float]:
    values = [float(token) for token in line.split()[:3]]
    if len(values) != 3:
        raise ValueError(f"Expected a 3D vector, got: {line!r}")
    return values


def is_integer_line(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    try:
        [int(token) for token in tokens]
    except ValueError:
        return False
    return True


def default_element_names(count: int) -> list[str]:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    names: list[str] = []
    for index in range(count):
        prefix = alphabet[index % len(alphabet)]
        suffix = index // len(alphabet)
        names.append(prefix if suffix == 0 else f"{prefix}{suffix}")
    return names


def parse_poscar(path: Path) -> PoscarStructure:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 8:
        raise ValueError(f"POSCAR file is too short: {path}")

    raw_lattice = [parse_vector(lines[index]) for index in range(2, 5)]
    axis_scales, lattice = parse_scaling(lines[1], raw_lattice)

    index = 5
    possible_elements = lines[index].split()
    if is_integer_line(possible_elements):
        counts = [int(token) for token in possible_elements]
        elements = default_element_names(len(counts))
        index += 1
    else:
        elements = possible_elements
        counts = [int(token) for token in lines[index + 1].split()]
        index += 2

    if len(elements) != len(counts):
        raise ValueError(f"Element and count lengths do not match in {path}")

    if lines[index].lower().startswith("s"):
        index += 1

    coordinate_mode = lines[index].lower()
    index += 1
    coordinate_count = sum(counts)
    coordinate_lines = lines[index : index + coordinate_count]
    if len(coordinate_lines) != coordinate_count:
        raise ValueError(f"Expected {coordinate_count} coordinates in {path}, got {len(coordinate_lines)}")

    if coordinate_mode.startswith("d"):
        fractional_coordinates = [parse_vector(line) for line in coordinate_lines]
    elif coordinate_mode.startswith("c") or coordinate_mode.startswith("k"):
        inverse_lattice = inverse_3x3(lattice)
        fractional_coordinates = []
        for line in coordinate_lines:
            cartesian = apply_cartesian_scale(parse_vector(line), axis_scales)
            fractional_coordinates.append(multiply_row_vector_matrix(cartesian, inverse_lattice))
    else:
        raise ValueError(f"Unsupported coordinate mode {lines[index - 1]!r} in {path}")

    return PoscarStructure(
        path=path,
        elements=elements,
        counts=counts,
        lattice=lattice,
        fractional_coordinates=fractional_coordinates,
    )


def format_formula(elements: Sequence[str], counts: Sequence[int]) -> str:
    parts: list[str] = []
    for element, count in zip(elements, counts):
        parts.append(element if count == 1 else f"{element}{count}")
    return "".join(parts)


def calculate_vacuum_thickness(fractional_coordinates: Sequence[Sequence[float]], c_length: float) -> float:
    if not fractional_coordinates:
        return 0.0

    wrapped_c = sorted(coordinate[2] % 1.0 for coordinate in fractional_coordinates)
    if len(wrapped_c) == 1:
        return c_length

    gaps = [wrapped_c[index + 1] - wrapped_c[index] for index in range(len(wrapped_c) - 1)]
    gaps.append((wrapped_c[0] + 1.0) - wrapped_c[-1])
    max_gap = max(gaps)
    return max(0.0, max_gap * c_length)


def analyze_structure(structure: PoscarStructure, root: Path) -> dict[str, str | int | float]:
    lattice_a, lattice_b, lattice_c = structure.lattice
    return {
        "folder": structure.path.parent.relative_to(root).as_posix(),
        "poscar_path": structure.path.relative_to(root).as_posix(),
        "total_atoms": sum(structure.counts),
        "chemical_formula": format_formula(structure.elements, structure.counts),
        "cell_volume": abs(determinant_3x3(structure.lattice)),
        "surface_area": norm(cross(lattice_a, lattice_b)),
        "vacuum_thickness": calculate_vacuum_thickness(structure.fractional_coordinates, norm(lattice_c)),
    }


def find_poscar_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("POSCAR") if path.is_file() and path.parent != root)


def build_summary(root: Path, sort_by: str | None = None) -> list[dict[str, str | int | float]]:
    rows = [analyze_structure(parse_poscar(path), root) for path in find_poscar_files(root)]
    if sort_by:
        rows.sort(key=lambda row: row[sort_by])
    return rows


def write_summary_csv(rows: Sequence[dict[str, str | int | float]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            serialized_row: dict[str, str | int] = {}
            for column in CSV_COLUMNS:
                value = row[column]
                serialized_row[column] = f"{value:.6f}" if isinstance(value, float) else value
            writer.writerow(serialized_row)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze slab POSCAR files in subdirectories.")
    parser.add_argument(
        "--sort-by",
        choices=CSV_COLUMNS,
        help="Sort rows by one of the output CSV columns before writing slab_summary.csv.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path.cwd()
    rows = build_summary(root, sort_by=args.sort_by)
    write_summary_csv(rows, root / "slab_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
