"""The corpus records that meet a stated definition of "similar".

A question can say what similar means: "its formula contains Mo; every band gap
recorded on it lies within 0.30 eV of 1.80 eV; the corpus classifies it as 2D;
excluding MoS2 and every record that joins MoS2 as a phase". Every stated condition
is a filter on the corpus, applied before the embedding ranks anything; the
similarity tool then ranks only inside this set and reports its size.

Elements are read from the formula as the tokens `[A-Z][a-z]?`, the same way a
reader checks "its formula contains X".
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

from django.db.models import Q

from materials.models import Material, Property
from materials.services.provenance import IMPORTED_ONLY

_EL = re.compile(r"[A-Z][a-z]?")


def elements_of(formula: str) -> Set[str]:
    return set(_EL.findall(formula or ""))


def _formula_regex(token: str) -> str:
    """Return the formula regex for an element or organic-cation token.

    Elements use the recommendation tool's regex, which does not read the F of
    FA, the B of BA or the P of PEA as an element. Organic-cation tokens use a
    plain match, which also finds them inside a cation chain (FAMA, BAMA).
    """
    from materials.services.rag import _ORGANIC_CATION_TOKENS, element_regex
    if token in _ORGANIC_CATION_TOKENS:
        return rf"{re.escape(token)}([^a-z]|$)"
    return element_regex(token)


def _norm(formula: str) -> str:
    return re.sub(r"\s+", "", formula or "")


def gap_window(c: Dict[str, Any]):
    """Return (lo, hi, every) from bandgap_min/max or bandgap_near/within, or None."""
    lo, hi = c.get("bandgap_min"), c.get("bandgap_max")
    near, within = c.get("bandgap_near"), c.get("bandgap_within")
    if near is not None and within is not None:
        lo, hi = float(near) - float(within), float(near) + float(within)
        return lo, hi, True     # "every gap within D of V" -- a distance binds every value
    if lo is None and hi is None:
        return None
    return (float(lo) if lo is not None else float("-inf"),
            float(hi) if hi is not None else float("inf"),
            bool(c.get("every_recorded_gap")))


def describe(c: Dict[str, Any]) -> List[str]:
    out = []
    if c.get("lead_free"):
        out.append("classified lead-free" + (" (from the question)" if "lead_free" in c.get("_from_question", ()) else ""))
    if c.get("chemical_class"):
        out.append(f"classified {c['chemical_class']}"
                   + (" (from the question)" if "chemical_class" in c.get("_from_question", ()) else ""))
    if c.get("structure_type"):
        out.append(f"structure type {c['structure_type']}")
    if c.get("dimensionality"):
        out.append(f"dimensionality {c['dimensionality']}")
    if c.get("deposited_structure"):
        out.append("deposits atomic coordinates"
                   + (" (from the question)" if "deposited_structure" in c.get("_from_question", ()) else ""))
    if c.get("contains_all"):
        out.append("formula contains " + " and ".join(c["contains_all"]))
    if c.get("contains_any"):
        out.append("formula contains " + " or ".join(c["contains_any"]))
    if c.get("contains_only"):
        out.append("formula has no elements other than " + ", ".join(c["contains_only"]))
    w = gap_window(c)
    if w:
        lo, hi, every = w
        rng = (f"within {c['bandgap_within']} eV of {c['bandgap_near']} eV"
               if c.get("bandgap_near") is not None and c.get("bandgap_within") is not None
               else f"in [{'' if lo == float('-inf') else lo}, {'' if hi == float('inf') else hi}] eV")
        out.append(f"{'every' if every else 'at least one'} recorded band gap {rng}")
    if c.get("exclude_compositions"):
        out.append("excluding " + ", ".join(c["exclude_compositions"])
                   + " and records joining them as a phase")
    return out


def has_criteria(c: Dict[str, Any]) -> bool:
    return bool(describe(c))


def _excluded_ids(compositions: Iterable[str]) -> Set[int]:
    from materials.services.rag import _compound_ids
    out: Set[int] = set()
    for comp in compositions:
        comp = _norm(comp)
        if not comp:
            continue
        out |= set(_compound_ids(comp))
        # A record that joins the composition as one of its phases ("CsPbBr3 | X").
        for mid, f in (Material.objects.filter(formula__contains="|", formula__icontains=comp)
                       .values_list("id", "formula")):
            if any(_norm(p) == comp or set(_compound_ids(_norm(p))) & out
                   for p in f.split("|")):
                out.add(mid)
    return out


def matching_ids(c: Dict[str, Any]) -> Set[int]:
    """Ids of every imported record meeting every stated condition."""
    qs = Material.objects.filter(**IMPORTED_ONLY)
    if c.get("lead_free"):
        # The corpus's own classification; a record it leaves unclassified
        # (NULL) is not shown to be lead-free.
        qs = qs.filter(is_lead_free=True)
    if c.get("chemical_class"):
        qs = qs.filter(chemical_class__iexact=c["chemical_class"])
    if c.get("structure_type"):
        qs = qs.filter(structure_type__iexact=c["structure_type"])
    if c.get("dimensionality"):
        qs = qs.filter(dimensionality__iexact=c["dimensionality"])
    if c.get("deposited_structure"):
        from materials.services.rag import deposited_coordinates_q
        qs = qs.filter(deposited_coordinates_q())
    for el in c.get("contains_all") or []:
        qs = qs.filter(formula__regex=_formula_regex(el))
    if c.get("contains_any"):
        q = Q()
        for el in c["contains_any"]:
            q |= Q(formula__regex=_formula_regex(el))
        qs = qs.filter(q)
    w = gap_window(c)
    if w:
        lo, hi, every = w
        from materials.services.quality import eligible_properties
        # Eligible gaps only: a suspect value neither admits nor excludes a record (quality.py).
        gaps = eligible_properties(Property.objects.filter(created_by__isnull=True, name__iexact="bandgap",
                                                           value__isnull=False))
        inside = gaps.filter(value__gte=lo - 1e-9, value__lte=hi + 1e-9).values("material_id")
        qs = qs.filter(id__in=inside)
        if every:
            outside = gaps.filter(Q(value__lt=lo - 1e-9) | Q(value__gt=hi + 1e-9)).values("material_id")
            qs = qs.exclude(id__in=outside)
    rows = qs.values_list("id", "formula")
    only = set(c.get("contains_only") or [])
    ids = {mid for mid, f in rows
           if not only or (elements_of(f) and elements_of(f) <= only)}
    if c.get("exclude_compositions"):
        ids -= _excluded_ids(c["exclude_compositions"])
    return ids


def contradictions(c: Dict[str, Any]) -> List[str]:
    """List conditions that no formula can meet together.

    These are reported as a contradiction in the request rather than as an
    empty corpus result.
    """
    out = []
    only = set(c.get("contains_only") or [])
    if only:
        missing = [el for el in c.get("contains_all") or [] if el not in only]
        if missing:
            out.append(f"contains_all names {', '.join(missing)}, which contains_only "
                       f"({', '.join(sorted(only))}) does not allow")
        if c.get("contains_any") and not set(c["contains_any"]) & only:
            out.append("no element of contains_any is allowed by contains_only")
    if c.get("lead_free") and "Pb" in (c.get("contains_all") or []):
        out.append("lead_free together with a formula that must contain Pb")
    if c.get("lead_free") and c.get("contains_any") and set(c["contains_any"]) <= {"Pb"}:
        out.append("lead_free together with contains_any that only allows Pb")
    return out


def single_condition_counts(c: Dict[str, Any]) -> List[str]:
    """Count the records each stated condition admits on its own."""
    keys = [k for k in c if not k.startswith("_") and k not in (
        "bandgap_within", "bandgap_max", "every_recorded_gap")]
    lines = []
    for key in keys:
        alone = {key: c[key]}
        if key == "bandgap_near":
            alone["bandgap_within"] = c.get("bandgap_within")
        if key == "bandgap_min":
            alone["bandgap_max"] = c.get("bandgap_max")
            alone["every_recorded_gap"] = c.get("every_recorded_gap")
        if key in ("bandgap_near", "bandgap_min") or describe(alone):
            label = "; ".join(describe(alone)) or key
            lines.append(f"{label} alone: {len(matching_ids(alone))}")
    return lines


def recorded_gaps(ids: Iterable[int]) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    from materials.services.quality import eligible_properties
    for mid, v in (eligible_properties(Property.objects.filter(material_id__in=list(ids), created_by__isnull=True,
                                                               name__iexact="bandgap", value__isnull=False))
                   .values_list("material_id", "value")):
        out.setdefault(mid, []).append(round(v, 4))
    return {k: sorted(v) for k, v in out.items()}
