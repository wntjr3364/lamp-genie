"""Electron affinities and ionization energies a question states.

A question may give the levels a band-alignment answer must use ("TiO2 EA 4.20
eV, IE 7.40 eV; CsSnI3 EA 4.13 eV, IE 5.40 eV"). Those override the platform's
table for that answer, value by value; a layer the question does not state
keeps the table's value, and the output says which is which.

Read clause by clause: a clause carrying both an EA and an IE value belongs to
the layer named in it -- the last formula-shaped name before the first value
("TiO2 has EA ...", "TiO2 (EA ..., IE ...)", "TiO2: IE ..., EA ..."), or the name
after "for" ("use EA ... and IE ... for TiO2"). A name is a formula-shaped token
(a digit or two capitals: TiO2, ITO, PEDOT:PSS, Spiro-OMeTAD) or a layer the
platform's table knows; a capitalised word such as "Use" or "The" is not.
"""
from __future__ import annotations

import re
from typing import Dict

_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_VALUE = re.compile(
    r"\b(?P<kind>EA|IE|electron\s+affinity|ionization\s+energy|ionisation\s+energy)\b\s*"
    r"(?:\((?:EA|IE)\)\s*)?(?:=|:|of|is)?\s*(?P<v>-?\d+(?:\.\d+)?)\s*eV", re.IGNORECASE)
_NAME = re.compile(r"[A-Z(][A-Za-z0-9.()]*(?:[:\-][A-Za-z0-9.()]+)*")
_NOT_NAMES = {"EA", "IE", "CBM", "VBM", "CBO", "VBO", "DFT", "PBE", "ETL", "HTL", "ETM", "HTM", "PSC", "PSCs"}
#: A figure, table, equation or reference label is not a layer: "(Fig.2a)", "Ref.12".
_LABEL = re.compile(r"(?:Figs?|Refs?|Eqs?|Table|Tab|No|S)\.?\s*\d", re.IGNORECASE)


def _known_layers():
    from materials.services.analysis_tools import BandAlignmentCalculator as B
    return set(B.PRESET_MATERIALS) | set(B.PRESET_ALIASES)


def _clean(token: str) -> str:
    """Strip brackets and a trailing period that belong to the surrounding sentence."""
    token = token.rstrip(".")
    if token.startswith("(") and token.endswith(")") and token.count("(") == 1:
        token = token[1:-1]                  # "(TiO2)", "(Fig.2a)": the bracketed word itself
    while token.endswith(")") and token.count(")") > token.count("("):
        token = token[:-1]
    while token.startswith("(") and token.count("(") > token.count(")"):
        token = token[1:]
    return token


def _is_name(token: str, known) -> bool:
    if not token or token in _NOT_NAMES or _LABEL.match(token):
        return False
    return token in known or bool(re.search(r"\d", token)) or sum(c.isupper() for c in token) >= 2


def _names_in(text: str, known):
    return [t for t in (_clean(m) for m in _NAME.findall(text)) if _is_name(t, known)]


def _pairs(values):
    """Group consecutive EA/IE matches into pairs, one per layer, in the order written."""
    pair, out = {}, []
    for v in values:
        kind = "EA" if v.group("kind").lower().startswith(("ea", "electron")) else "IE"
        if kind in pair:              # a second EA before an IE: start again from it
            pair = {}
        pair[kind] = v
        if len(pair) == 2:
            out.append(pair)
            pair = {}
    return out


def levels_in(text: str) -> Dict[str, Dict[str, float]]:
    """Return {layer: {"electron_affinity": x, "ionization_energy": y}} stated in `text`.

    Each EA/IE pair belongs to the last name written before it since the
    previous pair ("TiO2 EA ..., IE ..., CsSnI3 EA ..., IE ..."); failing that,
    to a "for" name right after it ("EA ... and IE ... for TiO2"), else to a
    "for" name later in the stretch before the next pair.
    """
    known = _known_layers()
    out: Dict[str, Dict[str, float]] = {}
    text = (text or "").translate(_SUB)
    for clause in re.split(r";|\.(?=\s|$)|\n", text):
        pairs = _pairs(_VALUE.finditer(clause))
        prev_end = 0
        for i, pair in enumerate(pairs):
            start = min(v.start() for v in pair.values())
            end = max(v.end() for v in pair.values())
            nxt = min(v.start() for v in pairs[i + 1].values()) if i + 1 < len(pairs) else len(clause)
            after = clause[end:nxt]
            right = re.match(r"\s*\)?\s*,?\s*for\s+(" + _NAME.pattern + r")", after)
            before = _names_in(clause[prev_end:start], known)
            name, prev_end = None, end
            if before:
                name = before[-1]
            elif right and _is_name(_clean(right.group(1)), known):
                name, prev_end = _clean(right.group(1)), end + right.end()
            else:
                later = re.search(r"\bfor\s+(" + _NAME.pattern + r")", after)
                if later and _is_name(_clean(later.group(1)), known):
                    name = _clean(later.group(1))
            if name:
                out[name] = {"electron_affinity": float(pair["EA"].group("v")),
                             "ionization_energy": float(pair["IE"].group("v"))}
    return out
