"""Record-bound factual answers rendered from tool evidence, without a second LLM rewrite.

A renderer accepts only a tool's versioned contract and never searches another record for a
number that happens to occur in an answer. Explanatory and mixed-tool turns stay with the agent;
payloads without the contract version are not rendered.
"""
from __future__ import annotations

import decimal
import json
import math
import re
from typing import Optional

from django.db.models import Count, Max, Min
from django.db.models.fields.json import KT
from .evidence_presentation import (COUNT_DEFINITION, GROUP_REQUEST, is_nitride_count_comparison,
                                    label as display_label, note, scope_lines)

#: Names under which the inventory tool is recognized.
INVENTORY_TOOL_NAMES = ("database_inventory", "corpus_inventory")


EVIDENCE_GROUP_LIMIT = 16
ANSWER_HISTORY_LIMIT = 7500  # keeps an answer replayed as history under 8000 characters.
_FIELDS = ("material_id", "material__formula", "material__name", "material__data_source",
           "material__external_id", "name", "unit", "method", "reference", "temperature_k")


def lookup_contract(rows, *, material_id, temperature_k, dimensionality, data_source,
                    matched_as_typed, kind, clean, qualifications=(), quality=None, excluded=None):
    """Group the rows in one bounded aggregate query, keeping each value with its record and method.

    `kind` annotates the evidence class of a row (`rag.evidence_kind_expr`); a group holds rows of
    one class.
    """
    # The stated phase (`context.phase`) is part of a group, so a solution and a film value never
    # share one.
    grouped = list(rows.order_by().annotate(evidence_kind=kind, phase=KT("context__phase"))
                   .values(*_FIELDS, "evidence_kind", "phase")
                   .annotate(n=Count("id"), minimum=Min("value"), maximum=Max("value"))
                   .order_by(*_FIELDS, "evidence_kind", "phase")[:EVIDENCE_GROUP_LIMIT + 1])
    complete = len(grouped) <= EVIDENCE_GROUP_LIMIT
    evidence = []
    for row in grouped[:EVIDENCE_GROUP_LIMIT]:
        evidence.append({
            "record_id": row["material_id"],
            "formula": clean(row["material__formula"], max_len=200),
            "name": clean(row["material__name"], max_len=500),
            "source": clean(row["material__data_source"], max_len=80),
            "source_id": clean(row["material__external_id"], max_len=200),
            "property": row["name"], "unit": clean(row["unit"], max_len=50),
            "method": clean(row["method"], max_len=200),
            "reference": clean(row["reference"], max_len=200),
            "kind": row["evidence_kind"],
            "phase": clean(row["phase"], max_len=30) or None,
            "temperature_k": row["temperature_k"],
            "n": row["n"], "min": row["minimum"], "max": row["maximum"],
        })
    return {"version": 1, "groups": evidence, "groups_complete": complete,
            "scope": {"material_id": material_id, "requested_temperature_k": temperature_k,
                      "dimensionality": dimensionality, "data_source": data_source,
                      "matched_as_typed": matched_as_typed},
            "qualifications": list(qualifications),
            # Recorded objections, structured
            # (`rag._quality_objections`, `rag._excluded_from_measurements`).
            "quality": quality or None, "excluded": excluded or None}


def _objection_values(q, ko) -> str:
    """Return the observed value and bound of a recorded objection, from its public fields."""
    parts = []
    observed = q.get("observed_value")
    if isinstance(observed, (int, float)) and not isinstance(observed, bool):
        parts.append(("관측값 " if ko else "observed ") + _number(observed))
    elif isinstance(observed, str) and observed:
        parts.append(("관측값 " if ko else "observed ") + _cell(observed))
    if isinstance(q.get("bound"), (int, float)) and not isinstance(q.get("bound"), bool):
        parts.append(("기준 " if ko else "bound ") + _number(q["bound"]))
    if not parts:
        return ""
    return ": " + (_cell(q["metric"]) + " " if q.get("metric") else "") + ", ".join(parts)


def _cell(value) -> str:
    # Database labels are text, not Markdown, hyperlinks, or instructions.
    text = str(value if value is not None else "—").replace("\n", " ").replace("\r", " ")
    text = text.replace("\\", "\\\\")
    for char in "|`[]*_<>":
        text = text.replace(char, "\\" + char)
    return text


def _number(value) -> str:
    """Format with twelve significant digits in positional notation; printed values avoid scientific
    notation such as "6.9e-06", which readers misread."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("not a finite numerical observation")
    if value == 0:
        return "0"
    text = format(decimal.Context(prec=12).create_decimal(repr(value)), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _span(group) -> str:
    lo, hi = _number(group["min"]), _number(group["max"])
    if group["min"] > group["max"]:
        raise ValueError("reversed observation range")
    unit = (group.get("unit") or "").strip()
    unit = "" if unit.casefold() in ("", "dimensionless", "—", "-") else " " + _cell(unit)
    return (lo if group["min"] == group["max"] else f"{lo} – {hi}") + unit


def _plain_range(span, unit) -> str:
    return (_number(span["min"]) if span["min"] == span["max"] else f"{_number(span['min'])}–{_number(span['max'])}") + (
        "" if (unit or "").casefold() in ("", "dimensionless") else " " + _cell(unit))


def _codes(entries) -> str:
    return ", ".join(f"{_cell(e['code'])} {e['n']:,}" for e in entries or [])


def quality_sentences(contract, unit, ko) -> list:
    """State the recorded objections in the reader's words.

    Covers how many values carry a warning (kept in the range), how many suspect values are left out
    and their stored range, and composition-assigned gaps that are not measurements. Returns plain
    lines, never list items, so a follow-up filter over the answer's rows never reads one as a row.
    """
    out = []
    quality, excluded = contract.get("quality"), contract.get("excluded") or {}
    if quality and quality.get("n_flagged"):
        n = quality["n_flagged"]
        out.append((f"참고: 경고가 기록된 값 {n:,}개는 원출처 값 그대로 위 범위에 포함되어 있습니다 ({_codes(quality.get('codes'))})."
                     if ko else f"Note: {n:,} value{'s carry' if n != 1 else ' carries'} a recorded warning and "
                     f"{'are' if n != 1 else 'is'} kept in the range above as the source published "
                     f"{'them' if n != 1 else 'it'} ({_codes(quality.get('codes'))})."))
        span = quality.get("range_without_objection")
        if span and span.get("min") is not None:
            out.append((f"경고값을 빼면 범위는 {_plain_range(span, unit)}입니다 (n={quality['n_without_objection']:,})."
                        if ko else f"Without the warned values the range is {_plain_range(span, unit)} "
                        f"(n={quality['n_without_objection']:,})."))
    suspect = excluded.get("suspect")
    if suspect and not suspect.get("range"):
        # Suspect values stated for several phases have no single stored range.
        n = suspect["n"]
        out.append((f"참고: 의심값 {n:,}개는 위 범위와 개수에서 제외했습니다 ({_codes(suspect.get('codes'))})."
                    if ko else f"Note: {n:,} stored value{'s' if n != 1 else ''} flagged as suspect "
                    f"{'are' if n != 1 else 'is'} left out of the range and counts above ({_codes(suspect.get('codes'))})."))
        suspect = None
    if suspect:
        n = suspect["n"]
        out.append((f"참고: 의심값 {n:,}개(저장값 {_plain_range(suspect['range'], unit)})는 위 범위와 개수에서 제외했습니다 "
                    f"({_codes(suspect.get('codes'))})."
                    if ko else f"Note: {n:,} stored value{'s' if n != 1 else ''} flagged as suspect "
                    f"(stored {_plain_range(suspect['range'], unit)}) {'are' if n != 1 else 'is'} left out of the range "
                    f"and counts above ({_codes(suspect.get('codes'))})."))
    composition = excluded.get("composition_basis")
    if composition:
        n = composition["n"]
        out.append((f"참고: 출처가 조성으로부터 할당한 밴드갭 {n:,}개({_plain_range(composition['range'], 'eV')})는 "
                    "측정값이 아니므로 집계에서 제외했습니다."
                    if ko else f"Note: {n:,} band gap{'s' if n != 1 else ''} the source assigned from the composition "
                    f"({_plain_range(composition['range'], 'eV')}) {'are' if n != 1 else 'is'} not measurements and "
                    "left out of the counts."))
    return out


_EXPLANATION = re.compile(
    r"\b(?:why|explain|interpret|mechanism|recommend|suggest|suitable|reliable|trust|better|best)\b"
    r"|\bhow\b(?!\s+(?:many|much|high|low|large|small)\b)"
    r"|왜|이유|설명|해석|의미|원인|어떻게|원리|추천|적합|신뢰|믿|유망|더\s*좋|안정", re.I)

# A full-range lookup has not filtered or sorted its individual observations.
# Keep those requests on the agent's selection path rather than presenting an
# unfiltered table as the answer to a numeric window or a top-N request.
_LOOKUP_SELECTION = re.compile(
    r"\b(?:filter|sort|rank)\w*\b|\btop\s*\d+\b"
    r"|(?:[<>≤≥]=?|\b(?:above|below|under|over|between|at\s+least|at\s+most)\b)\s*[-+]?\d"
    r"|\d+(?:\.\d+)?\s*(?:[a-zA-Z/%°μµ]+)?\s*(?:미만|이하|이상|초과)|정렬|상위\s*\d|걸러", re.I)


#: A composition family ("Bi2Te3-based", "doped"): a selection over many records, not one lookup.
_FAMILY_WORDS = re.compile(r"계열|기반|도핑|패밀리|-based\b|\bbased\s+on\b|\bfamily\b|\bdoped\b", re.I)


def _asks_a_selection(question: str) -> bool:
    """Return True if the question asks for a ranking or a composition family.

    The full-range table answers neither, so the agent's own answer is kept: it holds the extremum,
    which the table's capped rows need not. Checked here because chat() also renders the agent's
    lookup runs, not only the direct route.
    """
    if _FAMILY_WORDS.search(question):
        return True
    from .rag import question_plan   # rag imports this module
    try:
        return bool(question_plan(question).of_kind("rank"))
    except Exception:   # an unreadable question is no reason to replace the agent's answer
        return True


def render_lookup_answer(question: str, runs) -> Optional[str]:
    """Render a factual, lookup-only turn; return None outside this contract.

    No model prose is kept beside the evidence: the lookup supplies every record-bound group, and a
    mixed calculation or an explanation continues through the normal agent path.
    """
    if (not runs or len(runs) > 8 or _EXPLANATION.search(question or "")
            or _LOOKUP_SELECTION.search(question or "") or _asks_a_selection(question or "")):
        return None
    ko = bool(re.search(r"[가-힣]", question or ""))
    sections, seen = [], set()
    try:
        if re.search(r"\b(?:calculate|compute|recompute)\b|계산\s*(?:해|하|할)", question or "", re.I):
            sections.append(("요청한 추가 계산은 실행되지 않았습니다. 아래는 조회값이며 계산 결과가 아닙니다. 서로 다른 기록·온도의 값이나 범위의 중앙값을 임의로 결합하지 않았습니다."
                             if ko else "The requested calculation was not performed. The values below are retrieved observations, not a calculated result; values from different records or temperatures have not been combined."))
        for run in runs:
            name = run.get("name") if isinstance(run, dict) else run.name
            p = run.get("payload") if isinstance(run, dict) else run.payload
            if (name != "property_lookup" or not isinstance(p, dict)
                    or p.get("lookup_unverified_identity") or p.get("lookup_additional_context")):
                return None
            contract = p.get("record_evidence") or {}
            if contract.get("version") != 1 or not contract.get("groups") or not p.get("by_unit"):
                return None
            key = json.dumps(p, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            scope = contract["scope"]
            head = f"{_cell(p['formula'])} · {_cell(display_label(p['resolved'], ko))}"
            sections.append(f"### {head}")
            sections.append((f"조회된 물성값 {p['n_measurements']}개 / 대상 물질 기록 {p['n_materials']}개."
                             if ko else
                             f"{p['n_measurements']} retrieved property values / {p['n_materials']} matching material records."))
            if scope.get("material_id") is not None:
                sections.append(("조회 대상 LAMP 기록 ID: " if ko else "Lookup LAMP record ID: ")
                                + str(scope["material_id"]))
            if scope.get("data_source") or scope.get("dimensionality"):
                sections.append(("조회 범위: " if ko else "Lookup scope: ") + ", ".join(
                    _cell(s) for s in scope_lines({k: scope[k] for k in ("data_source", "dimensionality") if scope.get(k)}, ko)))
            sections += (["\n단위별 전체 조회 범위 (서로 다른 방법·구조를 평균내지 않음):"] if ko else
                         ["\nFull lookup ranges by unit (not an average across methods or structures):"])
            for unit, group in p["by_unit"].items():
                sections.append(f"- {_span({**group, 'unit': unit})}; n={group['n']}")
            if p.get("by_source_range"):
                sections += ["", ("| 출처 | 해당 출처의 범위 | n |" if ko else "| Source | Source-specific range | n |"),
                             "|---|---|---|"]
                for source, groups in p["by_source_range"].items():
                    sections += [f"| {_cell(source)} | {_span(g)} | {g['n']} |" for g in groups]
                if p.get("sources_omitted"):
                    sections.append(("추가 출처 미표시 수: " if ko else "Further sources not displayed: ")
                                    + str(p["sources_omitted"]))
            sections += ["", ("| LAMP 기록 | 물질 / 시료명 | 출처 · 출처 ID | 물성 | 값 / 범위 | n | 온도 (K) | 방법 · 근거 유형 | 문헌 / DOI |"
                                if ko else
                                "| LAMP record | Material / sample | Source · source ID | Property | Value / range | n | Temperature (K) | Method · evidence type | Reference / DOI |"),
                         "|---|---|---|---|---|---|---|---|---|"]
            for row in contract["groups"]:
                label = row["formula"] or row["name"]
                if row["name"] and row["name"] != label:
                    label += " / " + row["name"]
                temp = _number(row["temperature_k"]) if row["temperature_k"] is not None else "—"
                if scope.get("material_id") is not None and row["record_id"] != scope["material_id"]:
                    return None
                sections.append("| " + " | ".join([
                    str(row["record_id"]), _cell(label), _cell(row["source"] + " · " + (row["source_id"] or "—")),
                    _cell(row["property"] + (f" ({row['phase']})" if row.get("phase") else "")), _span(row),
                    str(row["n"]), temp,
                    _cell((row["method"] or "unspecified") + " · " + row["kind"]), _cell(row["reference"] or "—")]) + " |")
            if not contract["groups_complete"]:
                sections.append(("\n표는 일부 출처·기록·방법별 그룹만 표시합니다. 위 단위별 범위와 개수는 전체 조회 결과입니다."
                                 if ko else "\nOnly some record/method groups are displayed; the ranges and counts above cover the full lookup."))
            if not scope["matched_as_typed"]:
                sections.append(("요청 문자열과 동일한 저장명이 아니라 정규화된 조성으로 연결했습니다. 표의 실제 저장명을 확인하세요."
                                 if ko else "The lookup matched a normalized composition, not an identical stored name; actual stored names are shown."))
            if scope.get("requested_temperature_k") is not None:
                requested = _number(scope["requested_temperature_k"])
                sections.append((f"요청 온도는 {requested} K입니다. 이 조회의 전체 범위는 해당 온도의 측정값이나 보간값을 뜻하지 않습니다. 표의 실제 온도를 확인하세요."
                                 if ko else f"Requested temperature: {requested} K. The overall range is not a measurement or interpolation at that temperature; use the actual temperatures in the table."))
            sections.extend(_cell(note(qualification, ko)) for qualification in contract.get("qualifications", []))
            units = list(p["by_unit"])
            sections.extend(quality_sentences(contract, units[0] if len(units) == 1 else "", ko))
        sections.append(("\nreported는 측정·계산 여부가 확인되지 않은 보고값입니다. 빈칸은 미기록이며, 같은 조성도 동일 구조·동일 시료임을 뜻하지 않습니다. 실행된 조회만 표시하며 추가 계산·물질 적합성은 판정하지 않았습니다."
                         if ko else "\n'reported' means the measurement/calculation class is unspecified. A blank is not stored. Equal compositions do not establish identical structures or samples. Only executed lookups are reported; no additional calculation or suitability verdict is implied."))
        return "\n".join(sections)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


_WINDOW_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*(?:<|>)")
_BOUND_RE = re.compile(r"(>=|>|<=|<)\s*(-?[\d.]+(?:e-?\d+)?)")
_CARRIED_WINDOW_RE = re.compile(r"^([A-Za-z_][\w-]*) with at least one recorded value in (-?inf|-?[\d.]+)-(-?inf|-?[\d.]+)")


def _windows_of(filters, carrying) -> list:
    """Return every window as (quantity, low, low inclusive, high, high inclusive).

    A key ("bandgap >= 2 and <= 3") and a carried window ("bandgap with at least one recorded value
    in 2-3 (edges included ...)") are read alike.
    """
    out = set()
    for key in filters:
        m = _WINDOW_KEY_RE.match(str(key))
        if not m:
            continue
        lo = hi = None
        lo_i = hi_i = False
        for op, value in _BOUND_RE.findall(str(key)):
            if op.startswith(">"):
                lo, lo_i = float(value), op == ">="
            else:
                hi, hi_i = float(value), op == "<="
        out.add((m.group(1), lo, lo_i, hi, hi_i))
    m = _CARRIED_WINDOW_RE.match(carrying or "")
    if m:
        lo = None if "inf" in m.group(2) else float(m.group(2))
        hi = None if "inf" in m.group(3) else float(m.group(3))
        out.add((m.group(1), lo, lo is not None, hi, hi is not None))
    return sorted(out, key=str)


def _same_count(payload) -> str:
    """Return the key under which two inventory results are the same answer.

    The key holds every output, the classification scope and the set of bounded quantities. A
    window written as a key ("bandgap >= 2 and <= 3") and the same window carried by the call
    ("bandgap with at least one recorded value in 2-3") bound the same quantity.
    """
    filters = dict(payload.get("filters") or {})
    carrying = str(filters.pop("carrying_property", "") or "")
    windows = _windows_of(filters, carrying)
    bounded = {w[0] for w in windows}
    if carrying and not _CARRIED_WINDOW_RE.match(carrying) and carrying not in bounded:
        windows.append((carrying, "present"))   # a carried quantity without a window
    scope = {k: v for k, v in filters.items() if not _WINDOW_KEY_RE.match(str(k))}
    definition = payload.get("nitride_definition") or {}
    return json.dumps([payload.get(k) for k in ("total", "operation_counts", "group_by", "groups", "complete",
                                                "distinct_formulas", "scoped_to", "device_records")]
                      + [scope, windows, {k: (v or {}).get("total") for k, v in definition.items()
                                                  if isinstance(v, dict)}], sort_keys=True, default=str)


def _distinct_inventory_runs(runs):
    """Keep one run per distinct answer, preferring one that states its windows as conditions."""
    kept, keys = [], []
    for run in runs:
        p = run.get("payload") if isinstance(run, dict) else run.payload
        name = run.get("name") if isinstance(run, dict) else run.name
        key = _same_count(p) if name in INVENTORY_TOOL_NAMES and isinstance(p, dict) else None
        if key is not None and key in keys:
            i = keys.index(key)
            earlier = kept[i].get("payload") if isinstance(kept[i], dict) else kept[i].payload
            if "with at least one recorded value" in str((earlier.get("filters") or {}).get("carrying_property", "")):
                kept[i] = run
            continue
        kept.append(run)
        keys.append(key if key is not None else object())
    return kept


def render_inventory_answer(question: str, runs) -> Optional[str]:
    """Keep the scoped record total separate from cross-group comparison counts."""
    if not runs or len(runs) > 8:
        return None
    try:
        runs = _distinct_inventory_runs(runs)
    except (AttributeError, TypeError):
        return None
    if _EXPLANATION.search(question or "") and not (
            COUNT_DEFINITION.search(question or "") and re.search(r"count|nitrid|질화물|개수|집계", question or "", re.I)
            and all(((r.get("payload") if isinstance(r, dict) else r.payload) or {}).get("nitride_definition") for r in runs)):
        return None
    ko = bool(re.search(r"[가-힣]", question or ""))
    sections = []
    try:
        for run in runs:
            name = run.get("name") if isinstance(run, dict) else run.name
            p = run.get("payload") if isinstance(run, dict) else run.payload
            if (name not in INVENTORY_TOOL_NAMES or p.get("inventory_evidence_version") != 1
                    or p.get("counted") != "materials" or p.get("device_crosstab")):
                return None
            total = p["total"]
            if isinstance(total, bool) or not isinstance(total, int) or total < 0:
                return None
            operation_counts = p.get("operation_counts", {})
            sections += ["### " + ("데이터베이스 집계" if ko else "Database inventory")]
            if operation_counts:
                sections.append("요청 조건을 만족하는 기록 수:" if ko else "Records meeting the requested conditions:")
                for condition, count in operation_counts.items():
                    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
                        return None
                    sections.append(f"- {_cell(display_label(condition, ko))}: **" + (f"{count:,}" if count is not None else
                                    ("집계 불가" if ko else "not counted")) + "**")
            else:
                sections.append(f"조건에 맞는 물질 기록: **{total:,}건**. 고유 물질·화학식의 개수가 아닙니다."
                                if ko else f"Matching material records: **{total:,}**. Records sharing a composition are counted separately.")
            filters = {**p.get("filters", {}), **p.get("scoped_to", {})}
            sections.append(("적용 범위: " if ko else "Scope: ")
                            + (_cell("; ".join(scope_lines(filters, ko))) if filters else ("전체 코퍼스" if ko else "whole corpus")))
            if p.get("distinct_formulas") is not None and not operation_counts:
                sections.append(("해당 범위의 서로 다른 저장 화학식: " if ko else "Distinct stored formulas in that scope: ")
                                + f"{p['distinct_formulas']:,}")
            definition = p.get("nitride_definition")
            if str(filters.get("chemical_class", "")).casefold() == "nitride" or definition:
                sections.append(("여기서 질화물은 플랫폼의 Nitride 분류입니다. 이 분류는 인식된 음이온 계열이 N뿐인 조성을 기준으로 하며, N과 다른 음이온 계열을 함께 포함한 혼합 음이온 분류와 구분됩니다."
                                 if ko else "Here, nitride means the platform's Nitride class, based on N as the only recognized anion family. N-containing mixed-anion compositions form a separate population."))
            if definition and definition.get("version") == 1 and definition.get("same_other_conditions"):
                narrow = definition["classified"]
                broad = definition["nitrogen_containing"]
                pairs = [(condition, count, broad["operation_counts"].get(condition))
                         for condition, count in operation_counts.items()] if operation_counts else [("", narrow["total"], broad["total"])]
                for condition, n, b in pairs:
                    if n is None or b is None:
                        continue
                    if isinstance(b, bool) or not isinstance(b, int) or b < 0:
                        return None
                    if len(pairs) > 1:
                        sections.append(_cell(display_label(condition, ko)) + ":")
                    sections.append((f"같은 나머지 조건에서 Nitride 분류는 {n:,}건이고, 혼합 음이온 등을 포함해 N을 함유한 모든 조성으로 넓히면 {b:,}건입니다."
                                     if ko else f"With the same other conditions, the Nitride class gives {n:,} records; all N-containing compositions, including mixed-anion records, give {b:,}."))
                    if b > n:
                        sections.append((f"두 기준의 차이는 {b - n:,}건입니다."
                                         if ko else f"The two definitions differ by {b - n:,} records."))
            if p.get("group_by") == "property":
                sections.append(("한 기록이 여러 물성을 가질 수 있으므로 물성별 개수는 서로 겹칩니다."
                                 if ko else "Property counts overlap: one record can carry several properties."))
            show_groups = bool(GROUP_REQUEST.search(question or "")) and not is_nitride_count_comparison(question or "")
            if show_groups and operation_counts:
                sections.append("조건별 그룹 분포는 이번 집계에 포함되지 않았습니다." if ko else
                                "A group breakdown for each constrained count was not returned in this lookup.")
            elif show_groups:
                if p.get("scoped_to"):
                    sections.append("비교표에는 다른 그룹도 함께 표시됩니다. 위 개수는 지정한 범위만의 집계입니다." if ko else
                                    "The comparison table also includes other groups; the count above is for the specified scope only.")
                sections += ["", "| " + _cell(display_label(p["group_by"], ko)) + (" | 기록 수 |" if ko else " | Records |"), "|---|---|"]
                sections += [f"| {_cell(group or '(unset)')} | {n:,} |" for group, n in p["groups"].items()]
            if show_groups and not p.get("complete"):
                sections.append(("그룹 표는 일부만 표시되어 있습니다. 전체 조회 개수는 위 집계값입니다."
                                 if ko else "The group listing is partial; use the scoped total above, not a sum of displayed groups."))
            if p.get("device_records") is not None:
                sections.append(("별도의 장치 기록 수 (물리적 타당성 선별 전 원시 집계): " if ko else
                                 "Separate device-record count (raw, before plausibility screening): ") + str(p["device_records"]))
            sections.extend("- " + _cell(note(warning, ko)) for warning in p.get("inventory_warnings", []))
        return "\n".join(sections)
    except (KeyError, AttributeError, TypeError, ValueError):
        return None


def render_factual_answer(question: str, runs) -> Optional[str]:
    """Render a lookup, inventory or device answer from its tool contract, or return None.

    A mixed-tool answer is never partly replaced and no tool call is inferred. An answer over the
    history limit is reduced to its aggregates.
    """
    answer = (render_lookup_answer(question, runs) or render_inventory_answer(question, runs)
              or render_device_answer(question, runs))
    if answer is None or len(answer) <= ANSWER_HISTORY_LIMIT:
        return answer
    ko = bool(re.search(r"[가-힣]", question or ""))
    # Summarize aggregates rather than cut a row, DOI or qualification in half and present the
    # excerpt as a complete table. Full contracts remain in tool_results.
    lines = [("결과가 길어 집계만 요약했습니다. 기록별 상세 근거는 도구 결과에 보존되어 있습니다."
              if ko else "The result is long, so only aggregates are shown here. Record-level evidence is retained in the lookup details.")]
    if re.search(r"\b(?:calculate|compute|recompute)\b|계산\s*(?:해|하|할)", question or "", re.I):
        lines.append("추가 계산은 실행되지 않았습니다. 아래는 조회 집계이며 계산 결과가 아닙니다."
                     if ko else "The requested calculation was not performed. These are lookup aggregates, not a calculated result.")
    for run in runs:
        name = run.get("name") if isinstance(run, dict) else run.name
        p = run.get("payload") if isinstance(run, dict) else run.payload
        if name != "property_lookup":
            continue
        scope = p["record_evidence"]["scope"]
        lines.append(f"\n{_cell(p['formula'])} · {_cell(display_label(p['resolved'], ko))}: n={p['n_measurements']}, "
                     + (f"대상 물질 기록 {p['n_materials']}개" if ko else f"{p['n_materials']} matching material records"))
        if scope.get("material_id") is not None:
            lines.append(f"LAMP record: {scope['material_id']}")
        for unit, group in p["by_unit"].items():
            lines.append(f"- {_span({**group, 'unit': unit})}; n={group['n']}")
        if p.get("excluded"):
            lines.append("제외된 저장값은 위 집계에 포함되지 않습니다. 제외 사유는 도구 결과에 있습니다."
                         if ko else "Excluded stored observations are not included in these aggregates; exclusion reasons are retained with the lookup.")
    lines.append("집계 범위는 특정 온도의 값이나 동일 시료의 값이 아니며 보간·평균·추가 계산을 하지 않았습니다. 기록 ID·물성·온도로 범위를 좁혀 주세요."
                 if ko else "Aggregate ranges do not identify values at a requested temperature or on the same sample. No interpolation, cross-record average or additional calculation was performed. Narrow the request by record ID, property and temperature.")
    compact = "\n".join(lines)
    if len(compact) <= ANSWER_HISTORY_LIMIT:
        return compact
    return ("결과가 너무 길어 전체 답변을 표시하지 못했습니다. 기록 ID와 물성을 지정해 범위를 좁혀 주세요. 상세 근거는 도구 결과에 보존되어 있습니다."
            if ko else "The full evidence is too long to display. Narrow the request by record ID and property. Detailed evidence is retained with the lookup; this is not a complete answer or a calculated result.")


def render_device_answer(question: str, runs) -> Optional[str]:
    """Only the tool's eligible ranked rows, never its rejected transparency example."""
    if not runs or len(runs) > 8 or _EXPLANATION.search(question or ""):
        return None
    ko = bool(re.search(r"[가-힣]", question or ""))
    sections = []
    try:
        for run in runs:
            name = run.get("name") if isinstance(run, dict) else run.name
            p = run.get("payload") if isinstance(run, dict) else run.payload
            if (name != "device_summary" or p.get("device_evidence_version") != 1
                    or p.get("lookup_unverified_identity") or p.get("lookup_additional_context")):
                return None
            rank = p["ranking"]
            if rank["order"] not in ("asc", "desc") or not isinstance(rank["rows"], list):
                return None
            evidence = rank.get("evidence_rows")
            if evidence is None or len(evidence) != len(rank["rows"]):
                return None
            sections += ["### " + _cell(p["formula"] or "device records"),
                         ((f"전체 장치 기록 {p['n_devices']:,}건 중 조건에 맞는 {p['n_selected']:,}건에서 정렬했습니다."
                           if ko else f"Ranked among {p['n_selected']:,} qualifying device records out of {p['n_devices']:,} stored device records.")),
                         ("정렬 기준: " if ko else "Ranking: ") + _cell(display_label(rank["metric"], ko))
                         + ((" 높은 순" if rank["order"] == "desc" else " 낮은 순") if ko else
                            (", highest first" if rank["order"] == "desc" else ", lowest first")),
                         ("적용 조건: " if ko else "Conditions: ") + _cell("; ".join(note(c, ko) for c in rank["conditions"]) or "none"), ""]
            if rank["rows"]:
                for i, row in enumerate(evidence, 1):
                    values = ", ".join(f"{caption} {_number(row[key])} {unit}" for key, caption, unit in (
                        ("pce", "PCE", "%"), ("voc", "Voc", "V"), ("jsc", "Jsc", "mA/cm²"), ("ff", "FF", "")) if row.get(key) is not None)
                    sections.append(f"{i}. device {row['device_id']} · LAMP record {row['record_id']}: " + values)
                    sections.append("   " + _cell(row["source"] + " · " + (row["source_id"] or "—")))
                    sections.append(("   측정 조건: " if ko else "   Measurement conditions: ") + _cell(row["condition"] or ("미기록" if ko else "not recorded")))
                    sections.append(("   문헌: " if ko else "   Reference: ") + _cell(row["reference"] or ("미기록" if ko else "not recorded")))
                    flags = [] if row.get("projection", {}).get("module") else \
                        row.get("projection", {}).get("projection_flags", [])
                    for q in row.get("objections") or []:
                        sev = ({"suspect": "의심", "warning": "경고"} if ko else {}).get(q["severity"], q["severity"])
                        sections.append(("   품질 기록" if ko else "   Recorded objection") + f" ({_cell(sev)}, {_cell(q['code'])})"
                                        + _objection_values(q, ko))
                    if row.get("pce") is not None and row["pce"] < 1 and rank["metric"] in ("voc", "jsc"):
                        sections.append("   주의: 기록된 PCE가 1% 미만입니다. 이 행은 해당 지표의 순위를 보여 줄 뿐, 실용적인 고효율 소자 추천이 아닙니다."
                                        if ko else "   Caution: recorded PCE is below 1%. This is a metric ranking, not a recommendation of a practically efficient device.")
                    if "inconsistent_iv" in flags:
                        sections.append("   주의: 보고된 PCE와 같은 행의 J–V 지표로 계산한 값 사이에 차이가 있어 원측정 확인이 필요합니다."
                                        if ko else "   Caution: the reported PCE differs from the value implied by this row's J–V metrics; inspect the original measurements.")
                    if "mixed_basis" in flags:
                        sections.append("   주의: 지표가 서로 다른 측정 방식이나 스캔에서 선택되어 단일 측정값 묶음이 아닙니다."
                                        if ko else "   Caution: the metrics were selected from different measurements or scans, rather than one measurement.")
                    if rank["metric"] == "eqe":
                        sections.append(f"   EQE: {_number(row['rank_value'])}%")
            else:
                sections.append("조건을 만족하는 장치 기록이 없습니다." if ko else "No device record qualifies.")
            if rank["excluded_screen_rows"]:
                sections.append((f"선별 기준을 통과하지 못한 {rank['excluded_screen_rows']:,}건은 위 순위에 포함하지 않았습니다."
                                 if ko else f"{rank['excluded_screen_rows']:,} records failing the screen were excluded from this ranking."))
            if any(row.get("pce") is not None and row["pce"] < 1 for row in evidence) and rank["metric"] in ("voc", "jsc"):
                sections.append("효율 조건도 필요하면 PCE 하한을 지정해 다시 순위를 요청하세요. 1%는 위 저효율 안내 기준일 뿐, 이번 순위의 추가 제외 조건으로 사용하지 않았습니다."
                                if ko else "Specify a minimum PCE if efficiency should also constrain the ranking. The 1% threshold above is a caution label, not an additional exclusion rule.")
            sections.extend("- " + _cell(note(warning, ko)) for warning in rank["notes"])
            sections.append(("표시된 장치 행의 값만 보고합니다. 서로 다른 장치의 지표를 합쳐 한 장치의 성능으로 만들지 않았습니다."
                             if ko else "Each entry retains one device's own values; metrics from different devices are not combined."))
        return "\n".join(sections)
    except (KeyError, TypeError, AttributeError, ValueError):
        return None
