"""Select a composition from the corpus by what its A, B and X sites hold.

Answers "among the compositions in the corpus holding <ions on each site>, take the
one with the <lowest|highest> <statistic> recorded <property>". Runs as a
`screen_candidates` mode whose arguments the model fills in:

* members: imported records the corpus classifies as perovskite structures whose
  formula holds only the stated ions on each site -- exactly one of them on a
  site in `exactly_one_of`, one or more on a site in `any_subset_of`, every
  listed one elsewhere -- in ABX3 proportions (one A-site ion and three X-site
  ions per B-site ion). A formula without numeric proportions, or joining two
  phases with '|', is not a composition.
* records with the same composition -- equal per-B shares of the decimals as
  written, compared exactly, never rounded -- are one member, named by that
  exact composition (a 4-decimal label is shown beside it for display); its
  statistic is over every value recorded on any of them, from every source and
  method, and the output says so. Every member is listed with its n; one with
  fewer than `min_values` values does not qualify for selection, and the output
  says which.
* ties at the extremum select every tied member; an empty family is stated.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Tuple

A_SITE = ("PEA", "GA", "EA", "FA", "MA", "BA", "Cs", "Rb", "K", "Na")
B_SITE = ("Pb", "Sn", "Ge", "Bi", "Sb", "Cu", "Ag")
X_SITE = ("I", "Br", "Cl", "F")
_TOL = 1e-6


def parse_exact(formula: str) -> Optional[Dict[str, Decimal]]:
    """{ion: count as written, exactly} of a plain formula, or None when it is
    not one composition written with numeric proportions: a phase join ('|'),
    a parenthesised group or any placeholder character ("x", "-x", "δ") fails
    (`formula_utils.strict_composition_exact`, shared with rag)."""
    from materials.services.formula_utils import strict_composition_exact
    return strict_composition_exact(formula)


def _share_text(share: Fraction) -> Optional[str]:
    """A per-B share as an exact positional decimal, or None when it has no
    finite one (a denominator with a prime factor other than 2 and 5)."""
    if share == 1:
        return ""
    if share == 3:
        return "3"
    den, twos, fives = share.denominator, 0, 0
    while den % 2 == 0:
        den, twos = den // 2, twos + 1
    while den % 5 == 0:
        den, fives = den // 5, fives + 1
    if den != 1:
        return None
    k = max(twos, fives)                  # share = numerator * (10**k / den) / 10**k
    digits = str(share.numerator * (10 ** k // share.denominator))
    if k == 0:
        return digits
    digits = digits.rjust(k + 1, "0")     # integer string arithmetic: no context rounding
    whole, frac = digits[:-k], digits[-k:].rstrip("0")
    return whole + ("." + frac if frac else "")


def _sites(counts: Dict[str, float]):
    a = {k: v for k, v in counts.items() if k in A_SITE}
    b = {k: v for k, v in counts.items() if k in B_SITE}
    x = {k: v for k, v in counts.items() if k in X_SITE}
    other = set(counts) - set(a) - set(b) - set(x)
    return a, b, x, other


def member_key(formula: str, a_site, b_site, x_site, exactly_one_of=(),
               any_subset_of=()) -> Optional[Tuple[Tuple[str, Fraction], ...]]:
    """The exact composition key when `formula` belongs to the stated family,
    else None: (ion, share per B-site ion) for each site in A, B, X order, the
    shares exact fractions of the written decimals. Two formulas are one
    member only when these are equal -- FA0.1500001 and FA0.1500002 are two,
    RbGeBr3, RbGe1.0Br3 and Rb2Ge2Br6 one."""
    exact = parse_exact(formula)
    if not exact:
        return None
    a, b, x, other = _sites({k: float(v) for k, v in exact.items()})
    if other or not a or not b or not x:
        return None
    btotal = sum(b.values())
    if btotal <= 0:
        return None
    if abs(sum(a.values()) / btotal - 1) > _TOL or abs(sum(x.values()) / btotal - 3) > _TOL:
        return None
    for site, held, stated in (("A", a, a_site), ("B", b, b_site), ("X", x, x_site)):
        if not set(held) <= set(stated):
            return None
        if site in exactly_one_of:
            if len(held) != 1:
                return None
        elif site in any_subset_of:
            pass                              # one or more of the listed ions
        elif set(held) != set(stated):
            return None
    bexact = sum((Fraction(exact[ion]) for ion in b), Fraction(0))
    key = []
    for ions in (a, b, x):
        shares = {ion: Fraction(exact[ion]) / bexact for ion in ions}
        key.extend(sorted(shares.items(), key=lambda kv: (-kv[1], kv[0])))
    return tuple(key)


def exact_composition(key, formulas: Iterable[str]) -> str:
    """The member's identity as a formula: its exact per-B shares when every
    share has a finite decimal form, else the first of its formulas as
    written (also exact). Screening computes on this text."""
    parts = []
    for ion, share in key:
        text = _share_text(share)
        if text is None:
            return sorted(formulas)[0]
        parts.append(ion + text)
    return "".join(parts)


def rounded_label(key) -> str:
    """The member rounded to four decimals, for display only: distinct
    members may share it, so nothing is grouped or selected by it."""
    parts = []
    for ion, share in key:
        s = float(share)
        parts.append(ion + ("" if abs(s - 1) < _TOL else
                            ("3" if abs(s - 3) < _TOL else f"{s:.4f}".rstrip("0").rstrip("."))))
    return "".join(parts)


def select(a_site: Iterable[str], b_site: Iterable[str], x_site: Iterable[str],
           exactly_one_of: Iterable[str] = (), order: str = "lowest", statistic: str = "median",
           prop: str = "bandgap", structure_type: str = "Perovskite", min_values: int = 1,
           any_subset_of: Iterable[str] = ()) -> dict:
    from materials.models import Material, Property
    from materials.services.provenance import IMPORTED_ONLY

    a_site, b_site, x_site = list(a_site), list(b_site), list(x_site)
    exactly_one_of = [s.upper() for s in exactly_one_of]
    any_subset_of = [s.upper() for s in any_subset_of]
    qs = Material.objects.filter(**IMPORTED_ONLY, structure_type__iexact=structure_type)
    for ion in b_site if len(b_site) == 1 and "B" not in exactly_one_of + any_subset_of else []:
        qs = qs.filter(formula__contains=ion)
    records, formulas = defaultdict(list), defaultdict(set)
    for mid, formula in qs.values_list("id", "formula"):
        key = member_key(formula, a_site, b_site, x_site, exactly_one_of, any_subset_of)
        if key:
            records[key].append(mid)
            formulas[key].add(formula)
    values = defaultdict(list)
    ids = [mid for group in records.values() for mid in group]
    # Eligible values only (quality.py): a suspect value is not part of a member's statistic.
    from materials.services.quality import eligible_properties
    for mid, v in (eligible_properties(Property.objects.filter(
            material_id__in=ids, created_by__isnull=True, name__iexact=prop,
            value__isnull=False)).values_list("material_id", "value")):
        values[mid].append(v)
    agg = {"median": statistics.median, "min": min, "max": max}[statistic]
    members = []
    need = max(1, min_values)
    for key, mids in records.items():
        vals = [v for mid in mids for v in values[mid]]
        members.append({"composition": exact_composition(key, formulas[key]),
                        "label": rounded_label(key), "records": len(mids), "n": len(vals),
                        statistic: agg(vals) if vals else None, "min": min(vals) if vals else None,
                        "max": max(vals) if vals else None, "qualifies": len(vals) >= need})
    members.sort(key=lambda m: m["composition"])
    selected: List[str] = []
    ranked = [m for m in members if m["qualifies"]]
    if ranked:
        best = (min if order == "lowest" else max)(m[statistic] for m in ranked)
        selected = sorted(m["composition"] for m in ranked if abs(m[statistic] - best) < 1e-9)
    return {"members": members, "selected": selected, "order": order, "statistic": statistic,
            "property": prop, "min_values": need,
            "spec": {"a_site": a_site, "b_site": b_site, "x_site": x_site, "exactly_one_of": exactly_one_of,
                     "any_subset_of": any_subset_of, "structure_type": structure_type}}


def render(result: dict) -> str:
    from materials.services.rag import _format_measurement as fmt
    stat, prop = result["statistic"], result["property"]
    spec = result["spec"]

    def site(name, ions):
        how = ("exactly one of " if name in spec["exactly_one_of"] else
               "one or more of " if name in spec.get("any_subset_of", ()) else "")
        return how + " / ".join(ions) + f" on {name}"
    lines = [f"Selected in code from the corpus: compositions (records classified {spec['structure_type']}, "
             f"ABX3 proportions) holding {'; '.join(site(n, i) for n, i in (('A', spec['a_site']), ('B', spec['b_site']), ('X', spec['x_site'])))}; "
             f"records of one composition pooled; {stat} over every recorded {prop} value on them "
             f"(all sources and methods).", "",
             f"| composition | records | values | {stat} | range |", "|---|---|---|---|---|"]
    for m in result["members"]:
        # A member below the stated minimum is listed, and marked: it cannot be selected.
        label = m.get("label")
        name = (m["composition"] + (f" (≈ {label})" if label and label != m["composition"] else "")
                + ("" if m.get("qualifies", True) else
                   f" (fewer than {result['min_values']} values; not eligible)"))
        if m[stat] is None:
            lines.append(f"| {name} | {m['records']} | 0 | — | — |")
        else:
            lines.append(f"| {name} | {m['records']} | {m['n']} | {fmt(m[stat])} | "
                         f"{fmt(m['min'])}–{fmt(m['max'])} |")
    if not result["members"]:
        lines.append("| none | — | — | — | — |")
        lines.append("\nNo composition in the corpus meets the stated family; say so.")
    elif not result["selected"]:
        lines.append(f"\nThe family has members, but no member has at least {result['min_values']} recorded "
                     f"value(s) of {prop}; no composition can be selected. Say so.")
    else:
        lines.append(f"\nSELECTED: {', '.join(result['selected'])} ({result['order']} {stat} {prop})"
                     + (" -- a tie; report every tied composition." if len(result["selected"]) > 1 else ""))
    return "\n".join(lines)
