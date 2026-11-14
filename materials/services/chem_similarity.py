"""Chemistry-aware similarity scoring for materials retrieval.

Text-embedding cosine treats MAPbBr3 and MAPbI3 as near-identical because of
the shared 'MAPb' prefix; these scores use the chemistry instead:

1. Composition vector cosine (general inorganic)
2. Site-aware perovskite Jaccard (ABX3-family)

`conceptual_cousins` lists the single-site substitutions of a perovskite formula.
"""
from __future__ import annotations
import math
import re
from typing import Optional

from .formula_utils import coefficient, format_coefficient


# Multi-character organic cation tokens used in hybrid perovskites, longest
# first, so the element alternative does not split PEA/PPA into ``Pe`` + ``A``
# and lose the organic A-site.
_CATION_TOKENS = tuple(sorted(("FA", "MA", "EA", "BA", "PEA", "PPA"), key=len, reverse=True))
_CATION_ALT = "|".join(_CATION_TOKENS)
_TOKEN_PATTERN = re.compile(rf'({_CATION_ALT}|[A-Z][a-z]?)(\d*\.?\d*)')


def parse_formula(formula: str) -> dict[str, float]:
    """Parse formula into element counts. Handles organic cations (FA, MA).

    Examples:
        'CsPbI3'         -> {'Cs': 1, 'Pb': 1, 'I': 3}
        'MA0.5FA0.5PbBr3' -> {'MA': 0.5, 'FA': 0.5, 'Pb': 1, 'Br': 3}
        'Cs2AgBiCl6'     -> {'Cs': 2, 'Ag': 1, 'Bi': 1, 'Cl': 6}
    """
    if not formula:
        return {}
    # PerovskiteDB lists layered compositions separated by "|": use the first.
    formula = formula.split("|")[0].strip()
    matches = _TOKEN_PATTERN.findall(formula)
    counts: dict[str, float] = {}
    for elem, count in matches:
        if not elem:
            continue
        # `coefficient` tolerates a bare "." capture (text ending in a period).
        counts[elem] = counts.get(elem, 0) + coefficient(count)
    return counts


def composition_similarity(f1: str, f2: str) -> float:
    """Cosine similarity between normalised composition vectors.

    Returns 0.0 for non-overlapping formulas, 1.0 for identical compositions
    (up to scale). Sensitive to atomic ratios: MAPbBr3 vs MAPbI3 gives < 1.0
    because the halide differs.
    """
    v1 = parse_formula(f1)
    v2 = parse_formula(f2)
    if not v1 or not v2:
        return 0.0
    keys = set(v1) | set(v2)
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
    n1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    n2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


# Perovskite site dictionaries
_A_SITES = {"FA", "MA", "EA", "BA", "PEA", "Cs", "Rb", "K", "Na", "Li",
            "Ba", "Sr", "Ca"}
_B_SITES = {"Pb", "Sn", "Ge", "Bi", "Sb", "In", "Ag", "Cu", "Ti", "Zr",
            "Hf", "Fe", "Mn", "Ni", "Zn", "Nb", "Ta", "W", "Mo", "V", "Cr"}
_X_SITES = {"I", "Br", "Cl", "F", "O", "S", "Se", "Te", "N"}


def split_perovskite_sites(formula: str) -> tuple[dict, dict, dict]:
    """Split formula into A-site, B-site, X-site occupancy dicts.

    Returns ({A_elements}, {B_elements}, {X_elements}).
    Empty dicts if formula has no clear A/B/X separation.
    """
    elements = parse_formula(formula)
    a = {e: c for e, c in elements.items() if e in _A_SITES}
    b = {e: c for e, c in elements.items() if e in _B_SITES}
    x = {e: c for e, c in elements.items() if e in _X_SITES}
    return a, b, x


def jaccard(d1: dict, d2: dict) -> float:
    """Weighted Jaccard: sum of minimum occupancies over sum of maximum occupancies."""
    if not d1 and not d2:
        return 1.0
    if not d1 or not d2:
        return 0.0
    keys = set(d1) | set(d2)
    inter = sum(min(d1.get(k, 0), d2.get(k, 0)) for k in keys)
    union = sum(max(d1.get(k, 0), d2.get(k, 0)) for k in keys)
    return inter / union if union > 0 else 0.0


def perovskite_site_similarity(f1: str, f2: str) -> float:
    """Site-aware Jaccard for ABX3-family perovskites.

    score = (J(A1,A2) + J(B1,B2) + J(X1,X2)) / 3

    For MAPbBr3 vs MAPbI3: A-match, B-match, X-mismatch -> 0.67
    For MAPbBr3 vs FAPbBr3: A-mismatch, B-match, X-match -> 0.67
    For MAPbBr3 vs MA0.5FA0.5PbBr3: A-partial, B-match, X-match -> ~0.83
    """
    a1, b1, x1 = split_perovskite_sites(f1)
    a2, b2, x2 = split_perovskite_sites(f2)
    if not (a1 and b1 and x1) or not (a2 and b2 and x2):
        return 0.0
    return (jaccard(a1, a2) + jaccard(b1, b2) + jaccard(x1, x2)) / 3.0


# Conceptual cousins in the perovskite family: "similar to MAPbBr3" means
# MAPbCl3, FAPbBr3, CsPbBr3 (single A/B/X-site substitutions), which a text
# embedding misses because it keys on the "MAPb" prefix. The substitution
# families below generate those candidates for a database lookup.

_A_FAMILIES = {
    # Monovalent organic + inorganic A-site cations (mutually substitutable)
    "monovalent_A": {"MA", "FA", "Cs", "K", "Rb", "Na", "EA", "BA"},
    # Divalent A-site (chalcogenide perovskites, oxide perovskites)
    "divalent_A": {"Ba", "Sr", "Ca"},
}
_B_FAMILIES = {
    # Divalent B-site for halide perovskites
    "halide_B": {"Pb", "Sn", "Ge"},
    # Tetravalent B-site for oxide/chalcogenide perovskites
    "oxide_B": {"Ti", "Zr", "Hf"},
    # Trivalent B-site for double perovskites
    "double_B": {"Bi", "Sb", "In", "Cr"},
}
_X_FAMILIES = {
    # Halides (mutually substitutable)
    "halide_X": {"I", "Br", "Cl", "F"},
    # Chalcogens (oxide/chalcogenide perovskites)
    "chalc_X": {"O", "S", "Se", "Te"},
}


def _format_abx3(a_dict: dict, b_dict: dict, x_dict: dict) -> str:
    """Format A/B/X site dicts as ABX3 formula string.

    Input dicts are relative occupancies, rescaled to a total of 1 (A, B) or
    3 (X).
    """
    def _site_str(d: dict, target_total: int) -> str:
        if not d:
            return ""
        cur_total = sum(d.values())
        if cur_total == 0:
            return ""
        scale = target_total / cur_total
        parts = []
        for elem, count in d.items():
            c = count * scale
            if abs(c - 1.0) < 1e-6:
                parts.append(elem)
            elif abs(c - round(c)) < 1e-6:
                parts.append(f"{elem}{int(round(c))}")
            else:
                # Positional rendering, so parse_formula reads the same value
                # back (see formula_utils.format_coefficient).
                parts.append(f"{elem}{format_coefficient(c)}")
        return "".join(parts)
    return _site_str(a_dict, 1) + _site_str(b_dict, 1) + _site_str(x_dict, 3)


def conceptual_cousins(formula: str, max_cousins: int = 12) -> list[str]:
    """Generate perovskite-family cousins by single A/B/X-site substitutions.

    For MAPbBr3: CsPbBr3, FAPbBr3, MAPbCl3, MAPbI3, MASnBr3, MAGeBr3, ...
    (same scaffold, one site changed).

    Limitation: double perovskites (Cs2AgBiBr6) and Ruddlesden-Popper hybrids
    with multi-cation sites are reduced to a single-site ABX3 scaffold, so
    their cousins are single-B variants rather than BB' pair substitutions.
    """
    a, b, x = split_perovskite_sites(formula)
    if not (a and b and x):
        return []

    # Identify which family each site belongs to
    def _find_family(elements: dict, families: dict) -> Optional[set]:
        for name, fam in families.items():
            if any(e in fam for e in elements):
                return fam
        return None

    a_fam = _find_family(a, _A_FAMILIES)
    b_fam = _find_family(b, _B_FAMILIES)
    x_fam = _find_family(x, _X_FAMILIES)
    if not (a_fam and b_fam and x_fam):
        return []

    # An ordered list rather than a set, so truncation to `max_cousins` is
    # deterministic across processes (set order follows per-process hashing).
    cousins: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        if candidate and candidate not in seen:
            seen.add(candidate)
            cousins.append(candidate)

    # Single-site swaps, each site group sorted and interleaved A -> B -> X so
    # the larger A-site family does not use up the whole budget.
    by_site = (
        [_format_abx3({new_a: 1}, b, x) for new_a in sorted(a_fam - set(a))],
        [_format_abx3(a, {new_b: 1}, x) for new_b in sorted(b_fam - set(b))],
        [_format_abx3(a, b, {new_x: 1}) for new_x in sorted(x_fam - set(x))],
    )
    for rank in range(max(len(group) for group in by_site)):
        for group in by_site:
            if rank < len(group):
                _add(group[rank])

    # Exclude the input itself.
    input_no_digits = re.sub(r"\d+$", "", formula)
    cousins = [c for c in cousins if c != formula and c != input_no_digits]
    return cousins[:max_cousins]
