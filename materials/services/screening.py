"""Choosing among named candidates by a computed quantity, in code.

"Of A, B, C and D, consider those whose band gap lies between 1.2 and 1.6 eV;
which has the widest?" is three operations: compute a quantity for every
candidate, keep the ones inside the window, take the extremum. Window membership
and an ordering are comparisons with one right answer, so they are done here
rather than in the language model's prose, over the values the platform's own
calculators return (the same functions `predict_bandgap`, `analyze_stability`,
`calculate_band_alignment` and `validate_formula` call); the model only reports
the result. The window and the ordering are whatever the caller passes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from materials.services.analysis_tools import (
    within_screening_window,
    BandAlignmentCalculator,
    BandgapPredictor,
    FormulaValidator,
    StabilityAnalyzer,
)

# quantity -> unit, and what computes it
QUANTITIES: Dict[str, str] = {
    "bandgap": "eV",
    "tolerance_factor": "",
    "octahedral_factor": "",
    "electron_affinity": "eV",
    "ionization_energy": "eV",
    "site_fraction": "",
    "conduction_band_offset": "eV",
    "valence_band_offset": "eV",
}
SITES = ("A", "B", "X")
_SITE_KEY = {"A": "a_site", "B": "b_site", "X": "x_site"}

# Values that differ by less than this are a tie: the calculators round to
# three decimals, so a smaller difference is not a difference they computed.
TIE = 5e-4


def _preset(name: str) -> Optional[Dict[str, Any]]:
    key = BandAlignmentCalculator.PRESET_ALIASES.get(name, name)
    row = BandAlignmentCalculator.PRESET_MATERIALS.get(key)
    return {"name": key, **row} if row else None


def criterion_label(quantity: str, element: Optional[str] = None,
                    site: Optional[str] = None) -> str:
    if quantity == "site_fraction":
        return f"{element} fraction on the {site} site"
    if quantity == "conduction_band_offset":
        return "ΔE_C"
    if quantity == "valence_band_offset":
        return "ΔE_V"
    return quantity.replace("_", " ")


def evaluate(candidate: str, stated_levels: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
    """Compute every quantity the platform's calculators give for one candidate.

    A quantity that cannot be computed (a double perovskite's band gap, a
    transport layer's tolerance factor, an electron affinity outside the preset
    table) is omitted from `values`, with the reason in `unavailable`.
    """
    from materials.services.rag import (
        _parse_perovskite_composition, abx3_calculation_refusal, perovskite_model_refusal,
    )

    values: Dict[str, Any] = {}
    unavailable: Dict[str, str] = {}
    detail: Dict[str, Any] = {}

    parsed = None
    refusal = None
    try:
        parsed = _parse_perovskite_composition(candidate)
        refusal = perovskite_model_refusal(parsed)
    except Exception as exc:  # an unparseable name is still a candidate
        refusal = f"not parsed as an ABX3 composition ({exc})"
    # Site fractions describe any formula as written; the ABX3 quantities need A 1, B 1, X 3.
    abx3_refusal = abx3_calculation_refusal(parsed) if parsed is not None and not refusal else refusal
    if parsed is not None and not refusal and abx3_refusal:
        for q in ("bandgap", "tolerance_factor", "octahedral_factor"):
            unavailable[q] = abx3_refusal
    if parsed is not None and not refusal and not abx3_refusal:
        bg = BandgapPredictor.predict(a_site=parsed["a_site"], b_site=parsed["b_site"],
                                      x_site=parsed["x_site"])
        if bg.get("error") or bg.get("bandgap") is None:
            unavailable["bandgap"] = bg.get("error") or "not computable"
        else:
            values["bandgap"] = bg["bandgap"]
            detail["bandgap"] = {k: bg.get(k) for k in
                                 ("bandgap", "linear_bandgap", "bowing_correction")}
        geo = StabilityAnalyzer.analyze_geometric(a_site=parsed["a_site"],
                                                  b_site=parsed["b_site"],
                                                  x_site=parsed["x_site"])
        if geo.get("error") or geo.get("tolerance_factor") is None:
            reason = geo.get("error") or "not computable"
            unavailable["tolerance_factor"] = unavailable["octahedral_factor"] = reason
        else:
            values["tolerance_factor"] = geo["tolerance_factor"]
            values["octahedral_factor"] = geo["octahedral_factor"]
            detail["stability"] = {
                "tolerance_factor": geo["tolerance_factor"],
                "octahedral_factor": geo["octahedral_factor"],
                "verdict": geo.get("stability"),
                "stable": within_screening_window(geo),
            }
    if parsed is not None and not refusal:
        for site in SITES:
            for element, frac in parsed[_SITE_KEY[site]].items():
                if frac:
                    values[f"site_fraction:{element}@{site}"] = round(frac, 6)
        detail["sites"] = {site: {e: round(f, 6) for e, f in parsed[_SITE_KEY[site]].items() if f}
                           for site in SITES}
    else:
        for q in ("bandgap", "tolerance_factor", "octahedral_factor", "site_fraction"):
            unavailable[q] = refusal or "not an ABX3 composition"

    preset = _preset(candidate)
    given = (stated_levels or {}).get(candidate) or {}
    if preset or len(given) == 2:
        values["electron_affinity"] = given.get("electron_affinity", (preset or {}).get("electron_affinity"))
        values["ionization_energy"] = given.get("ionization_energy", (preset or {}).get("ionization_energy"))
        detail["level_source"] = "stated" if given else "table"
    else:
        reason = "no electron affinity / ionization energy in the platform's table"
        unavailable["electron_affinity"] = unavailable["ionization_energy"] = reason

    fv = FormulaValidator.validate(candidate)
    if fv.get("valid"):
        charge = fv.get("charge") or {}
        detail["charge_balance"] = {"balanced": bool(fv.get("charge_balance")),
                                    "status": charge.get("status"),
                                    "total": charge.get("total"),
                                    "terms": charge.get("terms", [])}
    return {"candidate": candidate, "values": values, "unavailable": unavailable,
            "detail": detail}


#: The one site each modelled ion occupies in an ABX3 halide, so a site fraction
#: named by its element alone ("the tin fraction") needs no site argument.
SITE_OF = {**{e: "A" for e in ("FA", "MA", "EA", "BA", "PA", "Cs", "Rb", "K")},
           **{e: "B" for e in ("Pb", "Sn", "Ge", "Bi")},
           **{e: "X" for e in ("I", "Br", "Cl", "F")}}
OFFSETS = ("conduction_band_offset", "valence_band_offset")


def value_of(row: Dict[str, Any], quantity: str, element: Optional[str] = None,
             site: Optional[str] = None) -> Optional[float]:
    if quantity == "site_fraction":
        if row["unavailable"].get("site_fraction"):
            return None
        # An element absent from a parsed site has fraction 0 there.
        return row["values"].get(f"site_fraction:{element}@{site}", 0.0)
    return row["values"].get(quantity)


def _normalise_criterion(c: Dict[str, Any], *, has_reference: bool) -> Optional[str]:
    """Validate one criterion in place, filling the site its element implies.

    Returns None when valid, otherwise an error message.
    """
    q = c.get("quantity")
    if q not in QUANTITIES:
        return f"unknown quantity {q!r}; use one of {sorted(QUANTITIES)}"
    if q == "site_fraction":
        el = (c.get("element") or "").strip()
        if not el:
            return "site_fraction needs the element, e.g. element='Sn'"
        c["element"] = el
        if c.get("site") not in SITES:
            if el not in SITE_OF:
                return (f"site_fraction of {el!r}: pass site='A', 'B' or 'X' "
                        f"(the site is inferred only for {', '.join(sorted(SITE_OF))})")
            c["site"] = SITE_OF[el]
    if q in OFFSETS and not has_reference:
        return f"{q} needs reference_layer and interface (the other layer, and the interface as written)"
    return None


def _interface_position(reference_layer: str, interface: Optional[str]):
    """Return ('first' | 'second' | None, error) for the reference layer's side.

    'first' when the interface is written reference/candidate, 'second' when
    candidate/reference.
    """
    if not interface:
        return None, ("pass interface exactly as the question writes it, e.g. "
                      f"'{reference_layer}/absorber' or 'absorber/{reference_layer}'")
    parts = [p.strip() for p in str(interface).split("/")]
    if len(parts) != 2:
        return None, f"interface {interface!r} must name two layers separated by '/'"

    def same(a, b):
        norm = lambda x: BandAlignmentCalculator.PRESET_ALIASES.get(x, x).lower()  # noqa: E731
        return norm(a) == norm(b)
    if same(parts[0], reference_layer) and not same(parts[1], reference_layer):
        return "first", None
    if same(parts[1], reference_layer):
        return "second", None
    return None, f"interface {interface!r} does not contain the reference layer {reference_layer!r}"


def screen(candidates: Sequence[str], filters: Sequence[Dict[str, Any]] = (),
           rank_by: Optional[Dict[str, Any]] = None, order: Optional[str] = None,
           reference_layer: Optional[str] = None,
           interface: Optional[str] = None,
           stability_criteria: Optional[Dict[str, Any]] = None,
           stated_levels: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
    """Compute, filter (inclusive bounds) and rank the candidates.

    Args:
        filters: [{"quantity", "element"?, "site"?, "min"?, "max"?}], the
            conditions that restrict which candidates are considered.
        rank_by: {"quantity", "element"?, "site"?}.
        order: "largest" or "smallest"; required with `rank_by`, no default.
        reference_layer, interface: the other layer and the interface as written
            (e.g. "TiO2/absorber"); adds each candidate's band offsets against
            that layer, which can be filtered or ranked.
        stability_criteria: optional t_min/t_max/mu_min/mu_max overriding the
            default stability window.
        stated_levels: {layer: {"electron_affinity", "ionization_energy"}} given
            in the question; they override the platform's table value by value.
    """
    reference_layer = (reference_layer or "").strip() or None
    # The reference layer is the other side of the interface, never a candidate.
    seen, names = {reference_layer} - {None}, []
    for c in candidates:
        c = (c or "").strip()
        if c and c not in seen:
            seen.add(c)
            names.append(c)
    filters = [dict(f) for f in filters]
    rank_by = dict(rank_by) if rank_by is not None else None
    errors = []
    position = None
    if reference_layer:
        position, err = _interface_position(reference_layer, interface)
        if err:
            errors.append(err)
    for c in filters + ([rank_by] if rank_by is not None else []):
        e = _normalise_criterion(c, has_reference=bool(reference_layer))
        if e:
            errors.append(e)
    if rank_by is not None and order not in ("largest", "smallest"):
        errors.append("order is required with rank_by: 'largest' or 'smallest', as the question asks")
    if errors:
        return {"error": "; ".join(errors)}

    rows = [evaluate(n, stated_levels) for n in names]
    crit = {k: v for k, v in (stability_criteria or {}).items() if v is not None}
    for row in rows:
        st = row["detail"].get("stability")
        if st and crit:
            t, mu = st["tolerance_factor"], st["octahedral_factor"]
            st["stable"] = (crit.get("t_min", float("-inf")) - 1e-9 <= t <= crit.get("t_max", float("inf")) + 1e-9
                            and crit.get("mu_min", float("-inf")) - 1e-9 <= mu
                            <= crit.get("mu_max", float("inf")) + 1e-9)
            st["verdict"] = "by the stated criteria " + ", ".join(f"{k} {_plain(v)}" for k, v in crit.items())
    offsets = {}
    if reference_layer:
        for row in rows:
            name = row["candidate"]
            layers = [reference_layer, name] if position == "first" else [name, reference_layer]
            res = BandAlignmentCalculator.calculate(layers, overrides=stated_levels)
            iface = f"{layers[0]}/{layers[1]}"
            if res.get("offsets") and not res.get("missing_layers"):
                o = res["offsets"][0]
                offsets[name] = {"interface": iface, "cbo": o["cbo"], "vbo": o["vbo"], "type": o["type"]}
                row["values"]["conduction_band_offset"] = o["cbo"]
                row["values"]["valence_band_offset"] = o["vbo"]
            else:
                why = "no electron affinity / ionization energy for " + ", ".join(res.get("missing_layers", []))
                offsets[name] = {"interface": iface, "error": why}
                row["unavailable"]["conduction_band_offset"] = row["unavailable"]["valence_band_offset"] = why
    for row in rows:
        row["checks"] = []
        eligible = True
        for f in filters:
            v = value_of(row, f["quantity"], f.get("element"), f.get("site"))
            lo, hi = f.get("min"), f.get("max")
            if v is None:
                ok, why = False, "not computable"
            else:
                ok = (lo is None or v >= lo - 1e-9) and (hi is None or v <= hi + 1e-9)
                why = "inside" if ok else "outside"
            row["checks"].append({"label": criterion_label(f["quantity"], f.get("element"), f.get("site")),
                                  "value": v, "min": lo, "max": hi, "passes": ok, "why": why})
            eligible = eligible and ok
        row["eligible"] = eligible
        row["rank_value"] = (value_of(row, rank_by["quantity"], rank_by.get("element"), rank_by.get("site"))
                             if rank_by else None)

    selected: List[str] = []
    if rank_by is not None:
        pool = [r for r in rows if r["eligible"] and r["rank_value"] is not None]
        if pool:
            best = (max if order == "largest" else min)(r["rank_value"] for r in pool)
            selected = [r["candidate"] for r in pool if abs(r["rank_value"] - best) <= TIE]
    else:
        selected = [r["candidate"] for r in rows if r["eligible"]]
    stated = dict(stated_levels or {})
    return {"rows": rows, "selected": selected, "order": order,
            "stated_levels": stated,
            # Layers of this call whose levels came from the question.
            "stated_used": sorted(n for n in names + ([reference_layer] if reference_layer else [])
                                  if n in stated),
            "rank_label": (criterion_label(rank_by["quantity"], rank_by.get("element"), rank_by.get("site"))
                           if rank_by else None),
            "rank_unit": QUANTITIES.get(rank_by["quantity"], "") if rank_by else "",
            "offsets": {k: v for k, v in offsets.items() if k in selected},
            "tie": len(selected) > 1 and rank_by is not None}


def _plain(v: float) -> str:
    """Up to three decimals, trailing zeros dropped; never scientific notation."""
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _fmt(v: Optional[float]) -> str:
    return "—" if v is None else _plain(v)


def render(result: Dict[str, Any]) -> str:
    """Render the tool output: the screening table, then the selection."""
    if result.get("error"):
        return f"Screening not run: {result['error']}"
    rows = result["rows"]
    labels = [c["label"] for c in rows[0]["checks"]] if rows else []
    # The ranked quantity gets its own column only when no filter already shows it.
    rank_col = bool(result["rank_label"]) and result["rank_label"] not in labels
    head = ["candidate"] + labels + ([result["rank_label"]] if rank_col else []) + ["kept"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["candidate"]]
        for c in r["checks"]:
            mark = "" if c["passes"] else " (outside)" if c["why"] == "outside" else " (not computable)"
            cells.append(_fmt(c["value"]) + mark)
        if rank_col:
            cells.append(_fmt(r["rank_value"]))
        cells.append("yes" if r["eligible"] else "no")
        lines.append("| " + " | ".join(cells) + " |")
    out = ["Screened in code: every value from the platform's calculators, window bounds inclusive.", ""]
    if result.get("stated_used"):
        # Name the layers whose EA/IE came from the question.
        out.insert(1, "Electron affinities and ionization energies stated in the question were used for: "
                   + ", ".join(result["stated_used"]) + "; any other layer uses the platform's table.")
    out += lines
    for r in rows:
        if not r["eligible"] and any(c["why"] == "not computable" for c in r["checks"]):
            reasons = "; ".join(sorted(set(r["unavailable"].values())))
            out.append(f"- {r['candidate']}: not computable ({reasons})")
    out.append("")
    sel = result["selected"]
    if not sel:
        out.append("SELECTED: none -- no candidate satisfies every stated condition.")
        return "\n".join(out)
    if result["rank_label"]:
        what = f"the {result['order']} {result['rank_label']} among the kept candidates"
        if result["tie"]:
            out.append(f"SELECTED (tie, {what}): " + ", ".join(sel))
        else:
            out.append(f"SELECTED ({what}): {sel[0]}")
    else:
        out.append("KEPT (every stated condition holds): " + ", ".join(sel))
    by_name = {r["candidate"]: r for r in rows}
    for name in sel:
        d = by_name[name]["detail"]
        facts = []
        if "bandgap" in d:
            b = d["bandgap"]
            facts.append(f"Eg = {b['bandgap']} eV (Vegard {b['linear_bandgap']} eV, "
                         f"bowing {b['bowing_correction']:+.3f} eV)")
        if "stability" in d:
            s = d["stability"]
            facts.append(f"t = {s['tolerance_factor']}, μ = {s['octahedral_factor']}, "
                         f"predicted {'stable' if s['stable'] else 'unstable'} ({s['verdict']})")
        if "charge_balance" in d:
            cb = d["charge_balance"]
            terms = " ".join(f"{t['species']}{t['state']:+d}×{_fmt(t['count'])}" for t in cb["terms"])
            facts.append(f"charge-balanced as written: {'yes' if cb['balanced'] else 'no'}"
                         + (f" (net {_fmt(cb['total'])}; {terms})" if cb["terms"] else ""))
        ea = by_name[name]["values"].get("electron_affinity")
        if ea is not None:
            facts.append(f"EA = {ea} eV, IE = {by_name[name]['values']['ionization_energy']} eV")
        off = result["offsets"].get(name)
        if off:
            if off.get("error"):
                facts.append(f"{off['interface']}: {off['error']}")
            else:
                p, q = off["interface"].split("/")
                facts.append(f"{off['interface']}: ΔE_C = EA({q}) − EA({p}) = {off['cbo']:+.3f} eV, "
                             f"ΔE_V = IE({q}) − IE({p}) = {off['vbo']:+.3f} eV, {off['type']}")
        out.append(f"- {name}: " + "; ".join(facts))
    out.append("Report the SELECTED candidate and its values above exactly; the window and the "
               "ordering have already been applied.")
    return "\n".join(out)
