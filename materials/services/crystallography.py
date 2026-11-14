"""Deposited crystal structures read from `coordinates_json`.

Builds a pymatgen Structure from every coordinate schema a source deposits (site occupancies,
fractional or Cartesian coordinates, a lattice matrix or six lattice parameters) and reports the
interatomic contacts that decide whether and how a cell is drawn.
"""

from __future__ import annotations

import math
import re
from typing import Sequence


def _lattice_from_parameters(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> list[list[float]]:
    al, be, ga = (math.radians(float(v)) for v in (alpha, beta, gamma))
    sin_gamma = math.sin(ga)
    if abs(sin_gamma) < 1e-12:
        raise ValueError("degenerate lattice: gamma gives zero sine")
    cx = float(c) * math.cos(be)
    cy = float(c) * (math.cos(al) - math.cos(be) * math.cos(ga)) / sin_gamma
    cz_sq = float(c) ** 2 - cx**2 - cy**2
    if cz_sq <= 0:
        raise ValueError("degenerate lattice: non-positive c-axis height")
    return [
        [float(a), 0.0, 0.0],
        [float(b) * math.cos(ga), float(b) * sin_gamma, 0.0],
        [cx, cy, math.sqrt(cz_sq)],
    ]


def _element_symbol(raw: object) -> str:
    match = re.search(r"[A-Z][a-z]?", str(raw or ""))
    if not match:
        raise ValueError(f"invalid or missing element label: {raw!r}")
    return match.group(0)


def site_occupancies(site: dict) -> dict[str, float] | None:
    """{element: occupancy} for a site that deposits them, else None.

    The single definition of a partially occupied site for every reader of
    `coordinates_json`. It accepts a per-site scalar `occu`/`occupancy`, a
    `species` list and a `{element: occupancy}` species map. None means the
    site names one element at full occupancy, the ordinary case.
    """
    scalar = site.get("occu", site.get("occupancy"))
    if scalar is not None:
        try:
            occupancy = float(scalar)
        except (TypeError, ValueError):
            raise ValueError(f"site occupancy is not a number: {scalar!r}")
        if not math.isfinite(occupancy) or not 0 < occupancy <= 1:
            raise ValueError(
                f"site occupancy must be finite and within (0, 1]: {occupancy!r}")
        if occupancy == 1.0:
            return None
        element = _element_symbol(
            site.get("element") or site.get("label")
            or ((site.get("species") or [{}])[0]).get("element")
        )
        return {element: occupancy}
    species = site.get("species")
    if isinstance(species, dict):
        return _species_map_occupancies(species)
    return _species_occupancies(species)


def _species_map_occupancies(species: dict) -> dict[str, float] | None:
    """The corpus layout `{element: occupancy}`; None for one element at full occupancy."""
    occupancies: dict[str, float] = {}
    for raw_element, raw in species.items():
        try:
            occupancy = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"site occupancy is not a number: {raw!r}")
        if not math.isfinite(occupancy) or occupancy <= 0:
            raise ValueError(f"site occupancy must be finite and positive: {occupancy!r}")
        element = _element_symbol(raw_element)
        occupancies[element] = occupancies.get(element, 0.0) + occupancy
    if not occupancies:
        return None
    if len(occupancies) == 1 and next(iter(occupancies.values())) == 1.0:
        return None
    return occupancies


def site_element(site: dict):
    """The element a site names: `element` / `label`, or the one element of its `species`.

    `species` is the corpus layout `{element: occupancy}` or a list of `{element, ...}` entries.
    Returns None when the site names no element.
    """
    raw = site.get("element") or site.get("label")
    if raw:
        return raw
    species = site.get("species")
    if isinstance(species, dict) and species:
        return next(iter(species))
    if isinstance(species, list) and species and isinstance(species[0], dict):
        return species[0].get("element")
    return None


def site_point(site: dict):
    """(coordinates, cartesian) of a site, or None.

    The corpus layout is `xyz` with a `frac` flag saying whether the three numbers are fractional;
    without the flag `xyz` is Cartesian. Also accepted: `abc` and a `frac` list (fractional), and
    `coords` with a `cartesian` flag.
    """
    frac = site.get("frac")
    if site.get("xyz") is not None:
        return site["xyz"], frac is not True
    if site.get("abc") is not None:
        return site["abc"], False
    if frac is not None and not isinstance(frac, bool):
        return frac, False
    if site.get("coords") is not None:
        return site["coords"], bool(site.get("cartesian", False))
    return None


def _species_occupancies(raw_species: object) -> dict[str, float] | None:
    """The `species` list form of `site_occupancies`."""
    if not isinstance(raw_species, list) or len(raw_species) == 0:
        return None
    occupancies: dict[str, float] = {}
    for entry in raw_species:
        if not isinstance(entry, dict) or "occupancy" not in entry:
            return None
        try:
            occupancy = float(entry["occupancy"])
        except (TypeError, ValueError):
            raise ValueError(f"site occupancy is not a number: {entry!r}")
        if not math.isfinite(occupancy) or occupancy <= 0:
            raise ValueError(f"site occupancy must be positive and finite: {occupancy!r}")
        element = _element_symbol(entry.get("element") or entry.get("label"))
        occupancies[element] = occupancies.get(element, 0.0) + occupancy
    if len(occupancies) == 1 and abs(next(iter(occupancies.values())) - 1.0) < 1e-9:
        # Fully occupied by one element: an ordered site written the long way.
        return None
    return occupancies


def pymatgen_structure_from_payload(
    payload: dict,
    lattice_parameters: Sequence[float] | None = None,
):
    """Build a pymatgen Structure from every coordinates_json schema in LAMP."""
    import numpy as np
    from pymatgen.core import Lattice, Structure

    if not isinstance(payload, dict):
        raise ValueError("coordinates_json is not an object")
    sites = payload.get("sites") or []
    if not sites:
        raise ValueError("coordinates_json carries no sites")

    matrix = payload.get("lattice_matrix") or payload.get("lattice")
    if matrix is None and lattice_parameters is not None:
        if len(lattice_parameters) != 6 or any(v is None for v in lattice_parameters):
            raise ValueError("six finite lattice parameters are required")
        matrix = _lattice_from_parameters(*lattice_parameters)
    if matrix is None:
        raise ValueError("no lattice matrix or parameters available")
    lattice = Lattice(np.asarray(matrix, dtype=float))

    species: list[object] = []
    coordinates: list[Sequence[float]] = []
    cartesian_flags: list[bool] = []
    for site in sites:
        raw_species = site.get("species") or []
        occupancies = site_occupancies(site)
        if occupancies is not None:
            # Keep the deposited composition of a partially occupied site: this
            # reader also feeds the CIF that `visualize_structure` returns.
            # Collapsing a site to one element is left to whatever draws it.
            species.append(occupancies)
        else:
            species.append(_element_symbol(site_element(site)))
        point = site_point(site)
        if point is None:
            raise ValueError("site carries no recognised coordinate field")
        coordinates.append(point[0])
        cartesian_flags.append(point[1])

    # Pymatgen takes one global coordinate convention, so every site is
    # converted to Cartesian; this also handles a payload that mixes conventions.
    cartesian = []
    for coords, is_cartesian in zip(coordinates, cartesian_flags):
        vector = np.asarray(coords, dtype=float)
        cartesian.append(vector if is_cartesian else vector @ lattice.matrix)
    return Structure(lattice, species, cartesian, coords_are_cartesian=True)


#: Below this, two atoms of a deposit overlap: the shortest bond in any crystal (H-H, 0.74 A) is far
#: longer. OQMD deposits a few such cells (e.g. entry 1389663 at 0.004 A), most of them calculations
#: the source's own energy gate rejects.
MIN_DRAWABLE_DISTANCE_A = 0.5


def shortest_distance(structure) -> float | None:
    """Shortest interatomic distance of a periodic structure (with images); None for < 2 sites."""
    if len(structure) < 2:
        return None
    centers, points, images, distances = structure.get_neighbor_list(MIN_DRAWABLE_DISTANCE_A * 4)
    nonzero = [d for c, p, d in zip(centers, points, distances) if d > 1e-8 or c != p]
    return float(min(nonzero)) if nonzero else None


def overlapping_atoms(structure) -> float | None:
    """The shortest distance if two atoms are closer than `MIN_DRAWABLE_DISTANCE_A`, else None."""
    d = shortest_distance(structure)
    return d if d is not None and d < MIN_DRAWABLE_DISTANCE_A else None


#: Shorter than any bond but H-H (0.74 A): drawn, with a note (e.g. OQMD entry 1049604 at 0.569 A).
SHORT_CONTACT_A = 0.7


def structure_warnings(structure, material=None) -> list[str]:
    """Notes to show beside a drawn cell.

    Covers an interatomic contact between 0.5 and 0.7 A and a calculation the source's own energy
    gate rejected.
    """
    out = []
    d = shortest_distance(structure)
    if d is not None and MIN_DRAWABLE_DISTANCE_A <= d < SHORT_CONTACT_A:
        out.append(f"Two atoms of the deposited cell are {d:.3f} \u00c5 apart, shorter than a chemical bond; the "
                   "coordinates may be wrong.")
    if material is not None:
        from materials.models import PropertyQualityIssue
        if PropertyQualityIssue.objects.filter(property__material=material, severity="suspect",
                                               code="oqmd_record_energy_rejected").exists():
            out.append("The source's own energy gate rejected this calculation (its energies are excluded); the "
                       "cell is shown as deposited.")
    return out

