"""What a follow-up turn asks of the conversation before it, read in code.

Some turns are about the conversation rather than the corpus ("of the list, which are below 300 K?",
"what does Pb-free mean?", "oh, 1,700 perovskites!", "the drawing looks wrong"). Each has a shape
that can be read without the model:

- a numeric condition over the rows an earlier answer listed (`RowCondition`, `answer_rows`,
  `row_values`);
- a question asking what a term means (`is_definition_question`, `platform_definitions`);
- a user restating a count the agent gave (`restated_count`);
- an objection to the previous answer (`is_complaint`);
- a condition on records that deposit atomic coordinates (`asks_deposited_structure`,
  `three_d_structure`).

Nothing here reads the database; the service resolves and checks the rows these functions find.
"""
from __future__ import annotations

import re
import math
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_NUM = r"[-−+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?"


def _num(text: str) -> float:
    return float(text.replace(",", "").replace("−", "-"))


def plain_number(value: float) -> str:
    """A bound as a person writes it: 300, 1.5, 323.15 (never in exponent notation)."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _decimals(text: str) -> int:
    return max(0, -Decimal(text.replace(",", "").replace("−", "-")).as_tuple().exponent)


# Numeric conditions over earlier rows

#: The quantities a follow-up filter names, each with the unit its values are written in and the
#: words (a table header, the text before a number) that name it. Temperature first: "k < 300" and
#: "temperature (K) < 300" name the temperature, written with or without its unit.
QUANTITIES: Tuple[Tuple[str, str, str], ...] = (
    ("temperature", "K", r"(?i:온도|temperature|temp\.?|kelvin|켈빈)|(?<![A-Za-z])[TkK](?![A-Za-z])"),
    ("bandgap", "eV", r"(?i:band\s*-?\s*gaps?|bandgaps?|밴드\s*갭|(?<![A-Za-z])gap)|(?<![A-Za-z])E\s*g(?![a-z])"),
    ("ZT", "", r"(?<![A-Za-z])ZT(?![A-Za-z])|(?i:figure\s+of\s+merit)"),
    ("PCE", "%", r"(?<![A-Za-z])PCE(?![A-Za-z])|(?i:efficienc\w*|효율)"),
    ("Voc", "V", r"(?<![A-Za-z])V\s*oc(?![A-Za-z])"),
    # The other two device metrics.
    ("Jsc", "mA/cm²", r"(?<![A-Za-z])J\s*sc(?![A-Za-z])|(?i:short[-\s]circuit\s+current(?:\s+density)?|단락\s*전류)"),
    ("FF", "", r"(?<![A-Za-z])FF(?![A-Za-z])|(?i:fill\s*factor|필\s*팩터|충진율|충전율)"),
    ("Seebeck", "μV/K", r"(?i:seebeck|제벡)"),
    ("Tc", "K", r"(?<![A-Za-z])T\s*c(?![A-Za-z])|(?i:critical\s+temperature|임계\s*온도)"),
    ("HOMO", "eV", r"(?<![A-Za-z])HOMO(?![A-Za-z])"),
    ("LUMO", "eV", r"(?<![A-Za-z])LUMO(?![A-Za-z])"),
    # Other quantities written in eV, so "delta EST under 0.15 eV" is not read as a band-gap bound.
    ("delta_EST", "eV", r"(?i:delta\s*-?\s*E\s*_?\s*ST|Δ\s*E\s*_?\s*ST|singlet[-\s]+triplet)"),
    ("work_function", "eV", r"(?i:work\s*function|일함수)"),
    ("electron_affinity", "eV", r"(?i:electron\s*affinit\w*|전자\s*친화)"),
    ("ionization_energy", "eV", r"(?i:ioni[sz]ation\s*(?:energy|potential)|이온화)"),
    ("formation_energy", "eV/atom", r"(?i:formation\s*energy|형성\s*에너지)"),
    ("energy_above_hull", "eV/atom", r"(?i:above\s*(?:the\s*)?(?:convex\s*)?hull|e_?hull)"),
)
_QUANTITY_RE = {name: re.compile(words) for name, _, words in QUANTITIES}
_TEMPERATURE_WORD_RE = re.compile(r"온도|temperature|temp\.?|kelvin|켈빈", re.I)
_UNIT_OF = {name: unit for name, unit, _ in QUANTITIES}
#: A unit written after the number, and the quantity it implies when no word names one.
_UNITS = (("mA/cm²", r"mA\s*/\s*cm(?:²|2|\^2)?"), ("K", r"K(?![A-Za-z/·])|kelvins?|켈빈"), ("°C", r"°\s*C|℃"),
          ("meV/atom", r"meV\s*/\s*atom"), ("eV/atom", r"eV\s*/\s*atom"),
          ("meV", r"meV(?![/A-Za-z])"), ("eV", r"eV(?![/A-Za-z])"),
          ("%", r"%|퍼센트|percent"), ("μV/K", r"[μµu]V\s*/\s*K"), ("V", r"V(?![A-Za-z/])"))
_UNIT_ALT = "|".join(f"(?P<u_{i}>{p})" for i, (_, p) in enumerate(_UNITS))

_LO_STRICT = r"above|over|more\s+than|greater\s+than|higher\s+than|larger\s+than|exceeding|>(?!=)"
_LO_INCL = r"at\s+least|no\s+less\s+than|not\s+below|>=|≥|=>"
_HI_STRICT = r"below|under|less\s+than|lower\s+than|smaller\s+than|cooler\s+than|<(?!=)"
_HI_INCL = r"at\s+most|up\s+to|no\s+more\s+than|not\s+above|<=|≤|=<"
_PRE_RE = re.compile(rf"(?P<cmp>{_LO_INCL}|{_HI_INCL}|{_LO_STRICT}|{_HI_STRICT})\s*(?P<n>{_NUM})\s*(?:{_UNIT_ALT})?",
                     re.I)
#: Comparators written after the number, each with its side and inclusiveness. One table serves both
#: the pattern and the direction, so a word is never matched and then read the wrong way round.
#: Longest first.
_POST_WORDS: Tuple[Tuple[str, str, bool], ...] = (
    (r"(?:을|를)?\s*넘지\s*(?:않|못)\w*", "hi", True),
    (r"이상", "lo", True), (r"이하", "hi", True), (r"미만", "hi", False), (r"초과", "lo", False),
    (r"(?:을|를)?\s*넘(?:는|게|어|은|으면|었|긴)\w*", "lo", False),
    (r"(?:보다\s*)?아래\w*|(?:보다\s*)?밑\w*|(?:에\s*)?못\s*미치\w*", "hi", False),
    (r"보다\s*(?:더\s*)?(?:작|낮|적)\w*", "hi", False), (r"보다\s*(?:더\s*)?(?:크|큰|높|많)\w*", "lo", False),
    (r"or\s+(?:more|higher|above|greater|over)", "lo", True), (r"or\s+(?:less|lower|below|under)", "hi", True),
    (r"and\s+(?:above|up)", "lo", False), (r"and\s+below", "hi", False),
)
_POST_RE = re.compile(
    rf"(?P<n>{_NUM})\s*(?:{_UNIT_ALT})?\s*(?P<cmp>" + "|".join(w for w, _, _ in _POST_WORDS) + ")", re.I)


def _post_comparator(word: str) -> Tuple[bool, bool]:
    """(lower bound, inclusive) of a comparator written after its number, from `_POST_WORDS`."""
    for pattern, side, inclusive in _POST_WORDS:
        if re.fullmatch(pattern, word, re.I):
            return side == "lo", inclusive
    raise ValueError(f"not a post-number comparator: {word!r}")
_UNIT2 = r"(?P<unit2>K(?![A-Za-z/·])|meV\s*/\s*atom|eV\s*/\s*atom|meV|eV(?![/A-Za-z])|%|°\s*C|℃)"
_BETWEEN_RE = re.compile(
    rf"between\s+(?P<a>{_NUM})\s*(?:{_UNIT_ALT})?\s*(?:and|to|-|–)\s*(?P<b>{_NUM})\s*{_UNIT2}?"
    rf"|(?P<a2>{_NUM})\s*(?:K|eV|meV|%)?\s*(?:~|-|–|에서)\s*(?P<b2>{_NUM})\s*{_UNIT2.replace('unit2', 'unit3')}?\s*"
    r"(?:사이|범위|까지)", re.I)


@dataclass
class RowCondition:
    """A bound on one quantity, read from the user's words."""
    quantity: str
    unit: str
    lo: Optional[float] = None
    lo_incl: bool = False
    hi: Optional[float] = None
    hi_incl: bool = False
    words: str = ""

    def admits(self, value: float) -> bool:
        if self.lo is not None and (value < self.lo or (value == self.lo and not self.lo_incl)):
            return False
        if self.hi is not None and (value > self.hi or (value == self.hi and not self.hi_incl)):
            return False
        return True

    def bounds(self) -> List[float]:
        return [b for b in (self.lo, self.hi) if b is not None]

    def describe(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        name = self.quantity
        if self.lo is not None and self.hi is not None:
            return (f"{plain_number(self.lo)}{unit} {'<=' if self.lo_incl else '<'} {name} "
                    f"{'<=' if self.hi_incl else '<'} {plain_number(self.hi)}{unit}")
        if self.lo is not None:
            return f"{name} {'>=' if self.lo_incl else '>'} {plain_number(self.lo)}{unit}"
        return f"{name} {'<=' if self.hi_incl else '<'} {plain_number(self.hi)}{unit}"


def _unit_matched(m: "re.Match") -> str:
    for i, (unit, _) in enumerate(_UNITS):
        if m.groupdict().get(f"u_{i}"):
            return unit
    return ""


def _quantity_before(text: str, start: int) -> Optional[str]:
    """The quantity a word in the 30 characters before `start` names, nearest first."""
    window = text[max(0, start - 30):start]
    best, pos = None, -1
    for name, _, _ in QUANTITIES:
        for m in _QUANTITY_RE[name].finditer(window):
            # "Tc" over "T"; a temperature word beats the letter k standing for kelvin
            if m.end() > pos or (m.end() == pos and name == "Tc"):
                best, pos = name, m.end()
    return best


def _quantity_for(text: str, start: int, unit: str) -> Optional[str]:
    """The quantity a bound is on: the word before it when that word's unit fits the written unit,
    else the unit's own quantity ("ZT below 500 K" bounds the temperature of the ZT, not the ZT)."""
    word = _quantity_before(text, start)
    if not unit:
        return word
    if word == "FF" and unit == "%":
        return word   # "FF above 75%": a fill factor in percent, not a PCE
    families = {"°C": "K", "meV": "eV", "meV/atom": "eV/atom"}
    family = families.get(unit, unit)
    if word and families.get(_UNIT_OF[word], _UNIT_OF[word]) == family:
        return word
    return _quantity_of_unit(unit)


def parse_row_conditions(text: str) -> List["RowCondition"]:
    """Every numeric condition a follow-up states ("k < 300", "below 500 K", "1.5 eV or more", "ZT
    above 1"), a lower and an upper bound on one quantity joined. The quantity comes from a word
    before the bound or from the unit; a bare number with neither is not a condition on anything."""
    text = (text or "").replace("℃", "°C")
    found: List[RowCondition] = []
    taken: List[Tuple[int, int]] = []
    for m in _BETWEEN_RE.finditer(text):
        a, b = (m["a"], m["b"]) if m["a"] is not None else (m["a2"], m["b2"])
        unit = _unit_matched(m) or re.sub(r"\s", "", m.group("unit2") or m.group("unit3") or "")
        q = _quantity_for(text, m.start(), unit)
        c = q and _condition(q, unit, lo=(_num(a), True), hi=(_num(b), True), words=m.group(0))
        if c:
            found.append(c)
            taken.append(m.span())
    for pattern, post in ((_PRE_RE, False), (_POST_RE, True)):
        for m in pattern.finditer(text):
            if any(s <= m.start() < e for s, e in taken):
                continue
            cmp, n, unit = m["cmp"].lower(), _num(m["n"]), _unit_matched(m)
            q = _quantity_for(text, m.start(), unit)
            if not q:
                continue
            if post:
                lo, incl = _post_comparator(cmp)
            else:
                lo = bool(re.fullmatch(f"{_LO_INCL}|{_LO_STRICT}", cmp, re.I))
                incl = bool(re.fullmatch(f"{_LO_INCL}|{_HI_INCL}", cmp, re.I))
            c = _condition(q, unit, **({"lo": (n, incl)} if lo else {"hi": (n, incl)}), words=m.group(0))
            if c:
                found.append(c)
                taken.append(m.span())
    joined: List[RowCondition] = []
    for c in found:
        partner = next((j for j in joined if j.quantity == c.quantity and
                        ((j.lo is None and c.hi is None and c.lo is not None and j.hi is not None) or
                         (j.hi is None and c.lo is None and c.hi is not None and j.lo is not None))), None)
        if partner is not None:
            if c.lo is not None:
                partner.lo, partner.lo_incl = c.lo, c.lo_incl
            else:
                partner.hi, partner.hi_incl = c.hi, c.hi_incl
            partner.words = f"{partner.words}; {c.words}"
        else:
            joined.append(c)
    return joined


def parse_row_condition(text: str) -> Optional["RowCondition"]:
    found = parse_row_conditions(text)
    return found[0] if found else None


def _quantity_of_unit(unit: str) -> Optional[str]:
    return {"K": "temperature", "°C": "temperature", "eV": "bandgap", "meV": "bandgap", "%": "PCE",
            "μV/K": "Seebeck", "V": "Voc", "mA/cm²": "Jsc"}.get(unit)


def _condition(quantity, unit, *, lo=None, hi=None, words=""):
    """A condition in the quantity's own unit (°C and meV converted), or None when the written unit
    is not the quantity's ("below 300 K" is no band-gap condition)."""
    target = _UNIT_OF.get(quantity, unit)

    def conv(v):
        if unit == "°C" and target == "K":
            return round(v + 273.15, 2)
        if quantity == "FF" and unit == "%":
            return v / 100.0
        if (unit, target) in (("meV", "eV"), ("meV/atom", "eV/atom")):
            return v / 1000.0
        return v
    if unit and unit != target and (unit, target) not in (("°C", "K"), ("meV", "eV"),
                                                        ("meV/atom", "eV/atom")) \
            and not (quantity == "FF" and unit == "%"):
        return None
    lo_v, lo_i = (conv(lo[0]), lo[1]) if lo else (None, False)
    hi_v, hi_i = (conv(hi[0]), hi[1]) if hi else (None, False)
    if any(v is not None and not math.isfinite(v) for v in (lo_v, hi_v)):
        return None
    return RowCondition(quantity, target, lo_v, lo_i, hi_v, hi_i, words=words.strip())


# References to an earlier answer's rows

#: The question points at rows an earlier answer listed.
EARLIER_ROWS_RE = re.compile(
    r"\b(?:of|among|from|in)\s+(?:those|these|them|that\s+list|this\s+list|the\s+(?:list|table|above|listed|"
    r"previous|rows|results|ones\s+(?:above|listed)))\b|\bwhich\s+of\s+(?:those|these|them|the\s+listed)\b"
    r"|\b(?:ones|rows|materials|records|entries)\s+(?:you\s+)?(?:just\s+)?(?:listed|mentioned|showed|gave|named)\b"
    r"|\byou\s+(?:just\s+)?(?:listed|mentioned|showed|named)\b|\bthe\s+listed\b|\bof\s+the\s+listed\s+ones\b"
    r"|(?:리스트|목록)\s*(?:중|에서|에|의|안)|그\s*중|이\s*중|위(?:의|에서|에)\s|위에서|앞(?:의|에서)|표(?:에서|의|에)"
    r"|(?:말한|나열한|보여준|리스트업한|언급한|알려준|뽑은)\s*(?:것|거|물질|애)", re.I)
#: A weaker pointer ("the ones", "those") that counts only when nothing else in the question starts
#: a new search.
WEAK_EARLIER_RE = re.compile(r"것들|거들|것만|것은|건\s|\bthe\s+ones\b|\bwhich\s+ones\b|\bthose\b|\bthem\b", re.I)
NEW_SEARCH_RE = re.compile(
    r"\b(?:find|search|recommend|list\s+(?:all|every)|show\s+me\s+(?:all|every)|in\s+the\s+(?:database|corpus)"
    r"|again|try\s+again|re-?run)\b|다시"
    r"|찾아|추천|검색|데이터베이스에서", re.I)


def refers_to_earlier_rows(question: str, *, names_a_formula: bool = False) -> bool:
    q = question or ""
    if re.search(r"\bagain\b|\bre-?run\b|다시\s*(?:검색|찾아)", q, re.I):
        return False   # "Again: ..." is a new search, not a filter of the rows shown
    if EARLIER_ROWS_RE.search(q):
        return True
    return bool(WEAK_EARLIER_RE.search(q)) and not NEW_SEARCH_RE.search(q) and not names_a_formula


@dataclass
class AnswerRow:
    """One row an answer listed: a table row (with its header) or a list line."""
    text: str
    cells: Dict[str, str] = field(default_factory=dict)

    def label(self, limit: int = 170) -> str:
        shown = " | ".join(c for c in self.cells.values() if c) if self.cells else self.text
        shown = re.sub(r"\*\*|__|`", "", shown)
        shown = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", shown).strip()
        return shown if len(shown) <= limit else shown[:limit - 1] + "…"


_LIST_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")
_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}")


def answer_rows(answer: str, is_material_row: Callable[[str], bool]) -> List[AnswerRow]:
    """The material rows an answer listed, in order, without repeats: Markdown table rows keyed by
    their header, and bullet or numbered lines. `is_material_row` decides that a line names a
    material."""
    rows: List[AnswerRow] = []
    seen = set()
    header: Optional[List[str]] = None
    lines = (answer or "").split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if _SEPARATOR_RE.match(nxt):
                header = cells
                continue
            if _SEPARATOR_RE.match(stripped):
                continue
            id_cell = bool(header) and any(k < len(header) and _ID_HEADER_RE.search(header[k]) and re.fullmatch(r"\d+", c)
                                           for k, c in enumerate(cells))
            if not (is_material_row(stripped) or id_cell):
                continue
            keyed ={(header[k] if header and k < len(header) else f"col{k}"): c for k, c in enumerate(cells)}
            row = AnswerRow(stripped, keyed)
        else:
            header = None if stripped and not stripped.startswith("|") else header
            if not _LIST_LINE_RE.match(line) or not is_material_row(stripped):
                continue
            row = AnswerRow(stripped)
        # Keep decimal points, signs and source labels: 1.2 is not 12, and two
        # sources reporting the same number are still two evidence rows.
        key = re.sub(r"\s+", " ", row.text.casefold()).strip()
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


_VALUE_RE = {
    "K": re.compile(rf"(?P<n>{_NUM})\s*(?:K(?![A-Za-z/·])|kelvin)"),
    "°C": re.compile(rf"(?P<n>{_NUM})\s*(?:°\s*C|℃)"),
    "eV": re.compile(rf"(?P<n>{_NUM})\s*eV(?![/A-Za-z])"),
    "%": re.compile(rf"(?P<n>{_NUM})\s*%"),
    "V": re.compile(rf"(?P<n>{_NUM})\s*V(?![A-Za-z/])"),
    "μV/K": re.compile(rf"(?P<n>{_NUM})\s*[μµu]V\s*/\s*K"),
}
_NAMED_VALUE_RE = re.compile(
    rf"\s*[=:]?\s*(?:of\s+|is\s+|값\s*)?(?P<censor>[<>≤≥]=?\s*)?(?P<n>{_NUM})")

_ENERGY_UNIT = r"(?:meV|eV)(?:\s*/\s*atom)?(?![/A-Za-z])"
_ROW_UNIT = re.compile(_ENERGY_UNIT + r"|°\s*C|℃|[μµu]V\s*/\s*K|(?<![/\w])K(?![/\w])|%|(?<!\w)V(?!\w)")
_RANGE_CELL = re.compile(
    rf"^\s*(?P<lo>{_NUM})\s*(?:{_ENERGY_UNIT}|K|°\s*C|℃|%)?\s*"
    rf"(?:–|—|-|~|\bto\b)\s*(?P<hi>{_NUM})\s*(?:{_ENERGY_UNIT}|K|°\s*C|℃|%)?\s*$", re.I)
_OBSERVATION = re.compile(
    rf"{_NUM}\s*(?:{_ENERGY_UNIT}|K|°\s*C|℃|%|V)?"
    rf"(?:\s*(?:–|—|-|~|\bto\b|/)\s*{_NUM}\s*(?:{_ENERGY_UNIT}|K|°\s*C|℃|%|V)?)?")


def _cell_values(cell: str, header: str, condition: RowCondition) -> List[Tuple[float, int]]:
    """Read every value in a cell in the condition's unit, without inventing a point estimate."""
    text = re.sub(r"\*\*|`", "", cell).strip()
    # A censored observation or uncertainty is not an exact value. Keep the
    # row undecided rather than treating '< 1.5' as a measurement of 1.5.
    if re.search(r"[<>≤≥±]|\+\s*/\s*-", text):
        return []
    if len({re.sub(r"\s+", "", u).replace("℃", "°C") for u in _ROW_UNIT.findall(text)}) > 1:
        return []  # mixed units need explicit per-observation conversion
    explicit, declared = _ROW_UNIT.search(text), _ROW_UNIT.search(header)
    if not (explicit or declared) and re.search(r"(?:meV|eV)\s*/", text + " " + header):
        return []
    unit = (explicit or declared).group() if explicit or declared else condition.unit
    unit = re.sub(r"\s+", "", unit).replace("℃", "°C").replace("µ", "μ").replace("uV", "μV")
    if unit != condition.unit and (unit, condition.unit) not in (("meV", "eV"), ("°C", "K"),
                                                               ("meV/atom", "eV/atom")):
        return []
    interval = _RANGE_CELL.fullmatch(text)
    numbers = [interval["lo"], interval["hi"]] if interval else re.findall(_NUM, text)
    result = []
    for number in numbers:
        try:
            value, decimals = _num(number), _decimals(number)
        except (InvalidOperation, ValueError, OverflowError):
            return []
        if not math.isfinite(value) or decimals > 18:
            return []
        if unit in ("meV", "meV/atom"):
            value, decimals = value / 1000, decimals + 3
        elif unit == "°C":
            value, decimals = round(value + 273.15, max(2, decimals)), max(2, decimals)
        result.append((value, decimals))
    return result


def row_values(row: AnswerRow, condition: RowCondition) -> List[Tuple[float, int]]:
    """(value, decimals written) of the condition's quantity in one row: the cell under a header
    that names the quantity, else every number written with the quantity's unit, else the number
    right after the quantity's name. Values in °C are converted to K for a temperature."""
    q, unit = condition.quantity, condition.unit
    name_re = _QUANTITY_RE.get(q)
    if row.cells:
        for head, cell in row.cells.items():
            if q == "temperature":
                # "Seebeck (μV/K)" is no temperature column: its K is a unit's denominator.
                head_names_it = bool(_TEMPERATURE_WORD_RE.search(head)) or bool(
                    re.search(r"(?<![/·\w])\(?\s*(?:T\s*)?\(?K\)?\s*\)?(?![\w/])|°C", head)
                    and not re.search(r"[/·]\s*K\b", head))
            else:
                head_names_it = bool(name_re and name_re.search(head))
            if q != "temperature" and q != "Tc" and _QUANTITY_RE["temperature"].search(head) \
                    and _TEMPERATURE_WORD_RE.search(head):
                head_names_it = False   # "ZT temperature (K)" is a temperature column, not ZT
            if head_names_it:
                return _cell_values(cell, head, condition)
        # Another column's eV value is not evidence for this quantity.
        return []
    text = row.text
    if name_re:
        named = name_re.search(text)
        after = _NAMED_VALUE_RE.match(text, named.end()) if named else None
        observation = _OBSERVATION.match(text, after.start("n")) if after else None
        if observation:
            # Do not strip an inequality, uncertainty, or unrecognised unit
            # suffix while extracting a short observation from a prose row.
            tail = text[observation.end():]
            if after["censor"] or re.match(
                    r"\s*(?:±|\+\s*/\s*-|/\s*[A-Za-z]|(?:meV|eV)\s*/)", tail):
                return []
            return _cell_values(observation.group(), "", condition)
        if named and q != "temperature":
            return []
        if any(pattern.search(text) for name, pattern in _QUANTITY_RE.items()
               if name != q and (_UNIT_OF[name] == unit
                                  or (unit.startswith("eV") and _UNIT_OF[name].startswith("eV")))):
            return []  # a named, different quantity is not an unnamed value of this one
    out: List[Tuple[float, int]] = []
    if unit:
        if re.search(r"[<>≤≥±]|\+\s*/\s*-", text):
            return []
        if unit == "eV":
            for m in re.finditer(rf"{_NUM}\s*(?:meV|eV)(?![/A-Za-z])", text):
                out.extend(_cell_values(m.group(), "", condition))
            return out
        for m in _VALUE_RE.get(unit, re.compile("$^")).finditer(text):
            out.append((_num(m["n"]), _decimals(m["n"])))
        if q == "temperature":
            for m in _VALUE_RE["°C"].finditer(text):
                out.append((round(_num(m["n"]) + 273.15, 2), _decimals(m["n"])))
    if not out and name_re:
        m = name_re.search(text)
        if m:
            after = _NAMED_VALUE_RE.match(text, m.end())
            if after:
                out.append((_num(after["n"]), _decimals(after["n"])))
    return out


# Definitions

_MEANS_RE = re.compile(
    r"\bwhat\s+(?:does|do)\b.{0,80}?\b(?:mean|stand\s+for|refer\s+to)\b|\bmeaning\s+of\b|\bdefinition\s+of\b"
    r"|\bdefine\b|\bwhat\s+is\s+meant\s+by\b|\bwhat\s+it\s+means\b|\bwhat\s+does\s+it\s+mean\b"
    r"|무슨\s*(?:뜻|의미)|(?:뜻|의미|정의)(?:가|는|이|를|은)?\s*(?:뭐|무엇|어떻게|알려)", re.I)
_WHAT_IS_RE = re.compile(r"^\s*what\s+(?:is|are)\s+(?:the\s+|a\s+|an\s+)?(?P<term>[^?.!]{1,40}?)\s*\??\s*$", re.I)
_KO_WHAT_IS_RE = re.compile(r"^\s*(?P<term>\S{1,20}(?:\s+\S{1,20}){0,2}?)\s*(?:이|가|란|이란|은|는)\s*"
                            r"(?:뭐야|뭔가요|뭐예요|뭐에요|뭐죠|뭐지|무엇인가요|무엇입니까|무엇이야|뭐니|\?)", re.I)
#: A question that wants records or values, whatever else it asks.
_DATA_CUE_RE = re.compile(
    r"\b(?:highest|lowest|largest|smallest|best|worst|most|least|top|how\s+many|which\s+(?:materials?|records?|"
    r"compounds?|ones?|sources?)|list|show|find|recommend|compare|records?\s+(?:of|for)|values?\s+(?:of|for))\b"
    r"|(?:above|below|over|under|between|<|>|≤|≥)\s*\d|\d\s*(?:미만|이하|이상|초과)"
    r"|몇\s*(?:개|종|건)|찾아|보여|추천|목록|리스트|가장|제일|비교", re.I)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[?!.])\s+|\n+", text or "") if s.strip()]


def is_definition_question(question: str, *, names_a_formula: bool = False) -> bool:
    """Does the turn ask what a term means ("What is the Pb-free? What does it mean that pb-free
    status is yes?", "What does dimensionality 2D mean here?") rather than for records? Never when
    it names a formula or asks for records or values."""
    if names_a_formula or _DATA_CUE_RE.search(question or ""):
        return False
    for sentence in _sentences(question):
        if _MEANS_RE.search(sentence):
            return True
        m = _WHAT_IS_RE.match(sentence) or _KO_WHAT_IS_RE.match(sentence)
        if m:
            term = m["term"]
            if len(term.split()) <= 4 and not re.search(r"\d(?![dD]\b)", term):
                return True
    return False


#: The platform's own definitions of the classifications a record carries. A definitional question
#: about one of them gets the platform's definition, not a search.
PLATFORM_TERMS: Tuple[Tuple[str, str], ...] = (
    (r"\b(?:pb|lead)[\s-]*free\b|무연|납\s*(?:없|프리|free)|lead[\s-]*free",
     "Pb-free (lead-free): for most imports a record's Pb-free flag is set from its formula alone -- yes when "
     "the formula contains no Pb, no when it contains Pb, blank when the composition is not known. "
     "PerovskiteDB records instead retain the provider's lead-free flag. It says nothing about other toxic "
     "elements, about performance, or about the other layers of a device; a search 'Pb-free' keeps only records "
     "flagged yes."),
    (r"\bdimensionalit\w*|\b[0-3]\s*-?D\b|차원",
     "dimensionality: the 0D/1D/2D/3D class a record carries, as its source labels it or as the importer "
     "classifies it (blank when there is no evidence). It describes the material's connectivity (a computed "
     "monolayer is 2D); it is not whether atomic coordinates are stored, and some sources label a layered bulk "
     "crystal 2D."),
    (r"structure[\s_-]*type|구조\s*유형",
     "structure type: the structure family a record is classified as (e.g. Perovskite, Layered (vdW), "
     "Metal-Organic Framework); blank when the source gives none. Crystal records of a perovskite composition "
     "from computed-crystal sources mostly carry no structure type, so 'perovskite' as a structure type is not "
     "every perovskite-composition record."),
    (r"structure\s+(?:information|data)|deposited\s+structure|atomic\s+coordinates|구조\s*정보|좌표",
     "deposited structure: a record deposits a structure when its source supplied atomic coordinates (a crystal "
     "cell) that LAMP stores; composition-only records carry none and cannot be drawn."),
    (r"energy\s+above\s+(?:the\s+)?(?:convex\s+)?hull|e_?hull",
     "energy above hull: the computed energy per atom above the convex hull of competing phases; 0 means on the "
     "computed hull (stable against decomposition in that calculation). It is not the formation energy."),
)


def platform_definitions(question: str) -> List[str]:
    return [text for pattern, text in PLATFORM_TERMS if re.search(pattern, question or "", re.I)]


# Objections and restated counts

COMPLAINT_RE = re.compile(
    r"\b(?:that(?:'s|\s+is)|this\s+is|it(?:'s|\s+is)|you(?:'re|\s+are))\s+(?:not\s+right|wrong|incorrect|"
    r"weird|strange|odd|off)\b|\blooks?\s+(?:wrong|weird|strange|odd|off|broken)\b|\bdoesn'?t\s+look\s+right\b"
    r"|\bnot\s+what\s+I\s+asked\b|\bwhy\s+did\s+you\s+say\b|\byou\s+said\b|\bseems?\s+(?:wrong|off|weird)\b"
    r"|\bthat\s+can'?t\s+be\s+right\b|\bnot\s+(?:right|correct)\b"
    r"|이상한데|이상해|이상하|틀렸|틀린|아닌데|아니잖|잘못\s*(?:됐|되었|된|나왔|알려|답|보여)|말이\s*안"
    r"|왜\s.{0,40}(?:없다고|있다고|했|하냐|하지|한\s*거|한거)", re.I)
#: The user correcting their own words ("sorry, I meant CsSnBr3") is no objection to the answer.
SELF_CORRECTION_RE = re.compile(r"잘못\s*말|제가\s*잘못|내가\s*잘못|\bI\s+meant\b|\bmy\s+(?:bad|mistake)\b|\bsorry,?\s+I\b",
                                re.I)
#: The rows the agent listed, pointed at ("the ones you listed").
AGENT_ROWS_RE = re.compile(r"(?:니가|네가|너가|당신이)\s*\S*\s*(?:리스트업|나열|말한|보여준|알려준|뽑은)"
                           r"|\byou\s+(?:just\s+)?(?:listed|mentioned|showed|gave|said)\b", re.I)


#: Doubting a value without calling it wrong: "Jsc is 0.105, isn't that strange?", "is that right?",
#: "really?".
DOUBT_RE = re.compile(r"\b(?:is|are)\b[^?.!]{0,50}\b(?:right|correct|accurate|plausible|real)\s*\?|\btoo\s+(?:high|low|large|small)\b"
                      r"|너무\s*(?:낮|높|크|작)|이상한\s*거\s*아니|맞아\s*\?|맞나|맞는\s*거|정말\s*\?|진짜\s*\?|\bis\s+that\s+(?:right|correct|possible|"
                      r"plausible)\b|\bisn'?t\s+(?:that|it)\s+(?:strange|odd|weird|wrong|too)\b|\breally\s*\?|\bhow\s+can\b"
                      r"|\bdoes\s+that\s+make\s+sense\b|\bsuspicious\b", re.I)
_ORDINAL_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
                  "ninth": 9, "tenth": 10, "첫": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8,
                  "아홉": 9, "열": 10}
_ORDINAL_RE = re.compile(r"(?P<n>\d{1,2})\s*(?:번째|번\s|위\s|등\s)|#\s*(?P<h>\d{1,2})\b|\b(?P<e>\d{1,2})(?:st|nd|rd|th)\b"
                         r"|\b(?:row|rank|number|no\.)\s*(?P<r>\d{1,2})\b"
                         r"|\b(?P<w>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b"
                         r"|(?P<k>첫|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*번째", re.I)


def is_challenge(question: str) -> bool:
    """The user questions a value or claim of an earlier answer (an objection or a doubt)."""
    return is_complaint(question) or bool(DOUBT_RE.search(question or ""))


def row_ordinal(question: str) -> Optional[int]:
    """The row a question points at by position ("the 4th one", "#4", "row 4"), or None."""
    m = _ORDINAL_RE.search(question or "")
    if not m:
        return None
    for g in ("n", "h", "e", "r"):
        if m.group(g):
            return int(m.group(g))
    return _ORDINAL_WORDS.get((m.group("w") or m.group("k") or "").lower())


def is_complaint(question: str) -> bool:
    q = question or ""
    return bool(COMPLAINT_RE.search(q)) and not SELF_CORRECTION_RE.search(q)


def points_at_listed_rows(question: str) -> bool:
    """An objection about, or a pointer to, the rows the agent listed (the earlier condition carries
    over)."""
    q = question or ""
    return bool(EARLIER_ROWS_RE.search(q)) and (
        is_complaint(q) or (bool(AGENT_ROWS_RE.search(q)) and not _ASKS_RE.search(q.strip())))


_COUNT_NOUN = (r"(?:\s*(?:여|여\s*개|개|종|가지|건|천)|\s+(?:\w+\s+){0,3}?(?:compounds?|materials?|species|kinds|"
               r"records?|entries|perovskites?|structures?|samples?|types?))")
_RESTATED_RE = re.compile(rf"(?:(?:over|more\s+than|about|around|nearly|almost|~|약|대략)\s*)?"
                          rf"(?P<n>\d{{1,3}}(?:,\d{{3}})+|\d{{2,}})(?P<noun>{_COUNT_NOUN})", re.I)
_ASKS_RE = re.compile(r"\?\s*$|^\s*(?:how|what|which|why|is|are|does|do|can)\b|몇|뭐|어떤|어디|왜|나요\s*\??$|까\s*\??$",
                      re.I)


def restated_count(question: str, earlier_answers: Sequence[str]) -> Optional[Tuple[str, int, str]]:
    """(the user's phrase, the earlier number, the earlier answer's line that gives it) when the
    user states a count back ("so there are more than 2,000 different perovskite compounds!") that
    an earlier answer gave within 20 %. A question asking for a count is not a restatement."""
    q = question or ""
    if _ASKS_RE.search(q.strip()):
        return None
    m = _RESTATED_RE.search(q)
    if not m:
        return None
    said = _num(m["n"])
    best = None
    for answer in earlier_answers:
        for line in (answer or "").split("\n"):
            for n in re.findall(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?![\d.])|(?<![\d.,])\d{2,}(?![\d.,])", line):
                v = _num(n)
                if v >= 10 and abs(v - said) <= 0.2 * max(v, said):
                    if best is None or abs(v - said) < abs(best[1] - said):
                        best = (m.group(0).strip(), int(v), line.strip())
        if best:
            break
    return best


def stated_count(question: str) -> Optional[str]:
    """The count a statement (not a question) asserts ("there are 5,000 MOFs in LAMP!"), or None."""
    q = question or ""
    if _ASKS_RE.search(q.strip()):
        return None
    m = _RESTATED_RE.search(q)
    return m.group(0).strip() if m else None


# Deposited structures

DEPOSITED_STRUCTURE_RE = re.compile(
    r"\b(?:with|having|has|have|that\s+(?:have|has)|carry(?:ing)?|including|deposit(?:s|ed|ing)?)\s+"
    r"(?:a\s+|an\s+|the\s+|any\s+)?(?:deposited\s+|stored\s+)?(?:crystal\s+|atomic\s+|3D\s+|3-D\s+|three-dimensional\s+)?"
    r"(?:crystal\s+)?(?:structures?(?!\s*[-_]?\s*types?\b)|coordinates|CIFs?|structural\s+(?:information|data))\b"
    r"|\b(?:structure|structural)\s+(?:information|data|files?)\b|\batomic\s+coordinates\b"
    r"|\b(?:3D|3-D|three-dimensional)\s+(?:crystal\s+)?structures?\b"
    r"|\bwhich\b.{0,60}\b(?:see|view|visuali[sz]e|draw|look\s+at)\b.{0,20}\b(?:crystal\s+)?structures?\b"
    r"|구조\s*(?:정보|데이터|파일)|원자\s*좌표|좌표\s*(?:가|이)?\s*있|(?:3d|3D|3차원)\s*구조|결정\s*구조\s*(?:가|이)\s*있", re.I)
#: "3D structure" in the coordinates sense, which the dimensionality reading would also claim.
THREE_D_STRUCTURE_RE = re.compile(r"\b(?:3D|3-D|three-dimensional)\s+(?:crystal\s+)?structures?\b|(?:3d|3D|3차원)\s*구조",
                                  re.I)


def asks_deposited_structure(text: str) -> bool:
    """Does the text restrict to records that deposit atomic coordinates ("perovskites with
    structure information", "similar materials that have a crystal structure", "3D structure")?"""
    return bool(DEPOSITED_STRUCTURE_RE.search(text or ""))


def three_d_structure(text: str) -> bool:
    return bool(THREE_D_STRUCTURE_RE.search(text or ""))


# References the answer cannot point to

_ID_MENTION_RE = re.compile(
    r"(?P<lead>\bLAMP\s*(?:record\s*|기록\s*|레코드\s*)?(?:id\s*)?[:#]?\s*|\b(?:record|레코드|기록)\s*(?:id\s*)?[:#]?\s*"
    r"|\bdevice\s*(?:id\s*)?[:#]?\s*|소자\s*(?:id\s*)?[:#]?\s*)(?P<id>\d{2,})(?![\d.,])", re.I)
_SOURCE_ID_RE = re.compile(r"(?<![\w:#/.-])(?:mp|mvc|JVASP|oqmd|2dm|paper)[-:_][A-Za-z0-9:_-]*\d[A-Za-z0-9:_-]*", re.I)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]+)[^)]*\)")
_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(\s*([^)\s]+)[^)]*\)")
_ID_HEADER_RE = re.compile(r"\bid\b|기록|레코드|record|device|소자", re.I)
_EMPTY_CELL_RE = re.compile(r"^\s*(?:—|–|-|n/?a|없음)?\s*$", re.I)


def _in_evidence(token: str, evidence: str) -> bool:
    if token.isdigit():
        return bool(re.search(rf"(?<![\d.]){token}(?![\d])", evidence))
    return token.lower() in evidence.lower()


def _url_supported(url: str, evidence: str) -> bool:
    if url.startswith("/") or url in evidence:
        return True
    doi = re.search(r"doi\.org/(10\.\S+)", url)
    return bool(doi and doi.group(1).rstrip(").,").lower() in evidence.lower())


def scrub_unsupported_references(answer: str, evidence: str) -> Tuple[str, List[str]]:
    """The answer with every record/device id and source identifier that no tool output (or earlier
    turn) of this conversation contains removed, made-up image and link URLs dropped, and table rows
    that are nothing but empty cells (a top-N padded with dashes) removed. Returns (answer, what was
    removed)."""
    removed: List[str] = []

    def _id(m):
        if _in_evidence(m.group("id"), evidence):
            return m.group(0)
        removed.append(m.group(0).strip())
        return m.group("lead") + "[id not in any tool output]"
    out = _ID_MENTION_RE.sub(_id, answer or "")

    def _src(m):
        if _in_evidence(m.group(0), evidence):
            return m.group(0)
        removed.append(m.group(0))
        return "[identifier not in any tool output]"
    out = _SOURCE_ID_RE.sub(_src, out)

    def _img(m):
        if _url_supported(m.group(1), evidence):
            return m.group(0)
        removed.append(m.group(1))
        return ""
    out = _IMAGE_RE.sub(_img, out)

    def _link(m):
        if _url_supported(m.group(2), evidence):
            return m.group(0)
        removed.append(m.group(2))
        return m.group(1)
    out = _LINK_RE.sub(_link, out)

    lines, header, padded = out.split("\n"), None, 0
    kept = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if _SEPARATOR_RE.match(nxt):
                header = cells
            elif not _SEPARATOR_RE.match(stripped) and header:
                if len(cells) > 1 and all(_EMPTY_CELL_RE.match(c) for c in cells[1:]):
                    padded += 1
                    continue
                changed = False
                for k, cell in enumerate(cells):
                    if k < len(header) and _ID_HEADER_RE.search(header[k]) and re.fullmatch(r"\d{2,}", cell) \
                            and not _in_evidence(cell, evidence):
                        removed.append(cell)
                        cells[k] = "—"
                        changed = True
                if changed:
                    line = "| " + " | ".join(cells) + " |"
        elif stripped:
            header = None
        kept.append(line)
    out = "\n".join(kept)
    if padded:
        removed.append(f"{padded} empty table row(s)")
        out += (f"\n\nNote: {padded} requested row(s) had no values in the tool output and were removed; this does not "
                "show that no further records exist.")
    if any("empty table row" not in r for r in removed):
        out += ("\n\nNote: identifiers or links that appear in no tool output of this conversation were removed from "
                "this answer.")
    return out, removed
