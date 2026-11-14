"""What a question states, read once and typed.

The tools apply conditions; this module only reads them from the user's own words, so that a tool
call which omits one can be given it back (`rag.py` merges, the question winning, as
`CellScreen.merge` does for cell conditions).

Rules this module keeps:
- Nothing is guessed. A phrase that names no stored quantity yields no condition; a number is
  never reinterpreted by its magnitude.
- Pasted text is cut out first (`own_text`): a pasted row's values, sources and instructions never
  become the user's conditions.
- Conditions bind to populations, not to sentences: one population per question unless the
  question contrasts distinct subjects; anaphora ("them", "those") and follow-ups resolve along
  the history, a same-kind condition replacing the earlier one and a new kind adding to it.
- Every bound carries its inclusivity ("above" is strict, "at least" is not), and every window its
  quantifier (any / every recorded value), aggregation (point / peak / min / median) and
  granularity (record / composition).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .source_names import CHAT_SOURCE_PATTERN


@dataclass(frozen=True)
class Bound:
    value: float
    inclusive: bool


@dataclass
class Condition:
    kind: str                      # window | rank | exact_formula | measured_only | temperature |
    #                                substitution | completeness | illumination
    target: Tuple[str, ...] = ()   # stored property names (alternatives) or device metric columns
    lo: Optional[Bound] = None
    hi: Optional[Bound] = None
    unit: str = ""
    direction: str = ""            # rank: asc | desc
    n: Optional[int] = None        # rank: how many asked for
    formulas: Tuple[str, ...] = ()
    substitution: Optional[Dict] = None
    completeness: str = ""         # recorded (all four present) | positive (all four > 0)
    irradiance_mw_cm2: Optional[float] = None
    quantifier: str = "any"        # any | every recorded value
    aggregation: str = "point"     # point | peak | min | median
    granularity: str = "record"    # record | composition
    qualitative: str = ""          # zero-gap | semiconducting
    routed: bool = False           # a computed quantity (screen_candidates), not a stored one
    span: Tuple[int, int] = (0, 0)
    population: Optional[int] = 0  # None: every population
    local: bool = False            # belongs to one COUNT operation, not to the population
    extra: tuple = ()              # a second endpoint written in another unit (parser-internal)
    op: Optional[int] = None       # COUNT operation a local condition belongs to
    op_parent: Optional[int] = None  # COUNT operation `op` is nested in ("how many of those")


@dataclass
class Population:
    sources: Tuple[str, ...] = ()
    formulas: Tuple[str, ...] = ()
    noun: str = ""
    span: Tuple[int, int] = (0, 0)


@dataclass
class Operation:
    kind: str                      # COUNT | SELECT | RANK | LOOKUP | COMPARE
    population: Optional[int] = 0
    conditions: List[int] = field(default_factory=list)
    span: Tuple[int, int] = (0, 0)
    nested_in: Optional[int] = None  # COUNT: index of the count this one is nested in


@dataclass
class PastedSpan:
    text: str
    directives: Tuple[str, ...] = ()


@dataclass
class QuestionPlan:
    own_text: str
    pasted: List[PastedSpan]
    populations: List[Population]
    conditions: List[Condition]
    operations: List[Operation]
    mentions: Tuple[str, ...] = ()
    unresolved: Tuple[str, ...] = ()

    def of_kind(self, kind: str) -> List[Condition]:
        return [c for c in self.conditions if c.kind == kind]


# Vocabulary

#: Stored quantities whose values the source states for a phase (`Property.context.phase`: solution
#: or film, Curated-OLED). A question naming the phase is answered from that phase only.
#: Each one's spellings (regex), beside its stored name: the recognizer below is built from this
#: table, so a quantity added here is recognized beside a phase word.
PHASE_BEARING_SPELLINGS = {
    "plqy": (r"plqys?", r"photoluminescence\s+quantum\s+yields?", r"양자\s*효율"),
    "pl_peak": (r"photoluminescence", r"(?<![a-z])pl(?![a-z])", r"emission", r"발광"),
    "pl_lifetime": (r"lifetimes?", r"decay\s+times?", r"수명"),
    "radiative_rate": (r"(?<![a-z-])radiative\s+(?:decay\s+)?rates?", r"(?<![a-z])kr(?![a-z])", r"(?<!비)복사\s*속도"),
    "nonradiative_rate": (r"non[-\s]?radiative\s+(?:decay\s+)?rates?", r"(?<![a-z])knr(?![a-z])", r"비\s*복사\s*속도"),
}
PHASE_BEARING = frozenset(PHASE_BEARING_SPELLINGS)
#: Latin words bounded by non-letters ("FilmOnly" is a name, not a phase); Hangul words take
#: particles.
_PHASE_WORDS = {"solution": r"(?<![A-Za-z])solutions?(?![A-Za-z])|용액",
                "film": r"(?<![A-Za-z])(?:thin[-\s]?)?films?(?![A-Za-z])|필름|박막"}
_PHASE_QUANTITY = "|".join(
    [re.escape(name).replace("_", "[_\\s-]?") for name in sorted(PHASE_BEARING_SPELLINGS, key=len, reverse=True)]
    + [spelling for spellings in PHASE_BEARING_SPELLINGS.values() for spelling in spellings])


def stated_phase(text: str) -> Optional[str]:
    """'solution' or 'film' when the text names one phase beside a phase-bearing quantity.

    Matches "solution PLQY", "PLQY in solution", "the PL peak of the film" and their Korean
    equivalents; None when the text names no phase or both.
    """
    text = text or ""
    # A question naming both phases anywhere asks about both ("radiative rates in solution and in
    # film"): no restriction, however far apart the words are.
    if all(re.search(words, text, re.I) for words in _PHASE_WORDS.values()):
        return None
    found = set()
    for phase, words in _PHASE_WORDS.items():
        before = rf"(?:{words})\W{{0,3}}(?:[\w가-힣]+\W+){{0,3}}?(?:{_PHASE_QUANTITY})"
        after = rf"(?:{_PHASE_QUANTITY})\W+(?:[\w가-힣]+\W+){{0,3}}?(?:{words})"
        if re.search(before, text, re.I) or re.search(after, text, re.I):
            found.add(phase)
    return found.pop() if len(found) == 1 else None


#: Names people write that the stored vocabulary does not carry, each with every stored name it
#: can mean. Declared, never inferred: a generic name keeps all its alternatives and a lookup
#: reports each under its own stored name; channels are never pooled. Longest first.
SEMANTIC_ALTERNATIVES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    # The corpus stores one PLQY and one PL peak per value, with the phase (solution, film) in the
    # property context; "solution PLQY" names the same stored quantity (its phase is shown with
    # each value).
    (r"solution\s+plqy|plqy\s+(?:in|of\s+the)\s+solution", ("plqy",)),
    (r"photoluminescence\s+quantum\s+yields?|\bplqys?\b|발광\s*양자\s*효율", ("plqy",)),
    (r"(?:pl|photoluminescence|emission)\s+(?:peaks?|maxim(?:um|a)|wavelengths?)|\bpl\b|발광\s*피크",
     ("pl_peak",)),
    # Common wordings that reach the stored names.
    (r"(?:carrier|pl|photoluminescence)\s+lifetimes?|캐리어\s*수명", ("pl_lifetime",)),
    (r"\bcb\s+(?:level|edge)s?\b|\bcbm\b|conduction[-\s]band\s+(?:minim(?:um|a)|levels?|edges?)",
     ("conduction_band_minimum",)),
    (r"\bvb\s+(?:level|edge)s?\b|\bvbm\b|valence[-\s]band\s+(?:maxim(?:um|a)|levels?|edges?)",
     ("valence_band_maximum",)),
    (r"(?:carrier\s+)?mobilit(?:y|ies)|이동도", ("carrier_mobility",)),
    (r"pore[-\s]?sizes?|pore\s+diameters?|\bpores?\b|cavit(?:y|ies)|구멍|기공", ("largest_cavity_diameter",
                                                                       "pore_limiting_diameter")),
    (r"effective\s+mass(?:es)?|유효\s*질량", ("electron_effective_mass", "hole_effective_mass")),
    (r"\bwf\b|work[-\s]?functions?|일함수", ("work_function",)),
    (r"밴드\s*갭|띠\s*간격", ("bandgap",)),
    # Korean and mixed wordings of energy above hull.
    (r"(?:에너지\s*)?above\s+(?:the\s+)?(?:convex\s+)?hull(?:\s*에너지)?|(?:hull|헐)\s*위\s*(?:의\s*)?에너지|"
     r"(?:hull|헐)\s*(?:기준\s*)?(?:초과\s*)?에너지", ("energy_above_hull",)),
    (r"제벡\s*계수|열기전력", ("Seebeck_coefficient",)),
    (r"열\s*전도도", ("thermal_conductivity",)),
    (r"전기\s*전도도", ("electrical_conductivity",)),
    # Resistivity only: the Korean word for electrical resistance names a resistance (ohm), which a
    # resistivity cannot give without the sample's geometry.
    (r"(?:전기\s*)?(?:비\s*)?저항\s*률|비\s*저항", ("electrical_resistivity",)),
    (r"파워\s*팩터|출력\s*인자", ("power_factor",)),
    (r"\bmass\s+densit(?:y|ies)\b", ("density",)),
    # A bare "gap" is the band gap ("gap 2-3 eV ... top 10 by gap"); "HOMO-LUMO gap", "optical
    # gap" and "singlet-triplet gap" are read whole first (an earlier, longer mention wins).
    (r"(?<![\w-])gaps?(?![\w-])", ("bandgap",)),
    (r"(?-i:(?<![\w.])E_?g(?![\w.]))", ("bandgap",)),  # "Eg 1.5-1.6 eV"
    (r"(?-i:(?<![\w.])PFs?(?![\w.]))", ("power_factor",)),  # "higher PF at room temp"
    # The Korean word for density on its own, never inside current, carrier, energy, charge, defect
    # or other compound densities.
    (r"(?<![가-힣])(?<!전류\s)(?<!캐리어\s)(?<!에너지\s)(?<!전하\s)(?<!전자\s)(?<!정공\s)(?<!결함\s)(?<!상태\s)"
     r"(?<!전력\s)(?<!광자\s)(?<!도핑\s)(?<!자속\s)(?<!입자\s)(?<!화소\s)밀도", ("density",)),
    (r"(?<!결함\s)(?<!결함)형성\s*에너지|formation\s+energ(?:y|ies)(?!\s+per)", ("formation_energy_per_atom",)),
    (r"체적\s*탄성\s*률|부피\s*탄성\s*률", ("bulk_modulus",)),
    (r"singlet[-\s]triplet\s+(?:gap|splitting)", ("delta_EST",)),
    (r"\btc\b|critical\s+temperatures?|(?<!glass\s)(?<!glass-)transition\s+temperatures?", ("superconducting_tc",)),
)

#: Device-record metrics (`Device` columns; EQE is the device's eqe_max property).
DEVICE_METRICS: Tuple[Tuple[str, str], ...] = (
    (r"power[-\s]conversion\s+efficienc(?:y|ies)|\bpces?\b|(?<![a-z])(?<!양자)(?<!양자\s)효율", "pce"),
    (r"open[-\s]circuit\s+voltages?|\bvocs?\b", "voc"),
    (r"short[-\s]circuit\s+current(?:\s+densit(?:y|ies))?|(?<![\w-])current\s+densit(?:y|ies)|\bjscs?\b|"
     r"(?:단락\s*)?전류\s*밀도", "jsc"),
    (r"fill\s+factors?|\bffs?\b", "ff"),
    (r"external\s+quantum\s+efficienc(?:y|ies)|\beqe(?:max|_max)?s?\b|외부\s*양자\s*효율", "eqe"),
)

_DEVICE_METRIC_NAMES = {metric for _, metric in DEVICE_METRICS}

#: Quantities whose value is itself a temperature: "Tc above 30 K" is a window on Tc.
TEMPERATURE_QUANTITIES = {"superconducting_tc", "transition_temperature"}
_TEMPERATURE_QUANTITY_RE = re.compile(
    r"\b(?:curie|n[ée]el|debye|melting|decomposition)\s+temperatures?\b", re.I)
#: A temperature quantity named right before a temperature range: the range is its value.
_TEMPERATURE_QUANTITY_BEFORE_RE = re.compile(
    r"\b(?:curie|n[ée]el|debye|melting|decomposition|critical|transition|glass[-\s]transition)\s+temperatures?"
    r"(?:\s+(?:is|of|in|within|in\s+the\s+range))?\s*$|\bt_?[cg]\s*$", re.I)

#: Quantities the platform computes rather than stores; a ranking on them is routed to
#: screen_candidates (`_parse_rank_request` already refuses them).
_COMPUTED_QUANTITY_RE = re.compile(
    r"tolerance\s+factor|octahedral\s+factor|\bmu\b|(?<![A-Za-z0-9])μ(?![A-Za-zΩ/])|band\s+offsets?|conduction[-\s]band\s+offset|"
    r"valence[-\s]band\s+offset|bowing|electron\s+affinit(?:y|ies)|ionization\s+energ|"
    # A composition's own proportions ("the highest caesium fraction on the A site", "the smallest
    # share"): read off the formula or the counts, not a stored property.
    r"\b[a-z]+\s+fraction\b|\bfraction\s+of\b|\bshare\b",
    re.I)

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "twenty": 20, "한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3,
                 "셋": 3, "네": 4, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
_NUM = (r"(?<![A-Za-z_])(?<![A-Za-z]-)(?:[-+−]?10\s*\^\s*[-+−]?\d+|[-+−]?10[⁺⁻−]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+|"
        r"[-+−]?(?:\d{1,3}(?:,\d{3})+(?![\d,])|\d+(?:[.,]\d+)?)"
        r"(?:\s*[x×X]\s*10\s*(?:\^\s*[-+−]?\d+|[⁺⁻−]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)|[eE][-+−]?\d+(?![A-Za-z]))?)")
_KNOWN_UNIT = (r"(?:(?:kilo|mega)-?electron[-\s]?volts?|milli-?electron[-\s]?volts?|electron[-\s]?volts?|밀리\s*전자\s*볼트|전자\s*볼트|percent|퍼센트|kelvin|켈빈|"
               r"나노미터|밀리볼트|볼트|마이크로미터|마이크론|옹스트롬|하트리|리드버그|a\.u\.|"
               r"(?:micro|nano|pico|milli)-?seconds?|"
               r"meV/atom|eV/atom|mV/K|µV/K|μV/K|uV/K|meV|eV|nm|Å|µm|um|ps|fs|ns|µs|μs|us|ms|%|K|°C|℃|mV|V|"
               r"mA\s*/\s*cm(?:2|²|\^2)|mW\s*/\s*cm(?:2|²|\^2)|W\s*/\s*\(?m\s*[·.]?\s*K\)?|S\s*/\s*(?:m|cm)|"
               r"cm(?:2|²)\s*/\s*V\s*s|m_e|GPa|MPa|g\s*/\s*cm(?:3|³|\^3)|g/cc|[µμ]B|Ω\s*[·.]?\s*m|ohm[-\s]?m|"
               r"W\s*/\s*m\s*K|K⁻¹|1/K|A\s*/\s*m(?:2|²))(?![A-Za-z0-9/_·^])")
#: Words that follow a number and are not a unit ("above 1 above 600 K", "1 eV and ..."); anything
#: else written there is kept as the unit, so an undeclared unit is refused rather than dropped.
_NOT_A_UNIT = {"and", "or", "in", "at", "for", "with", "from", "the", "of", "to", "on", "by", "as", "than", "per",
               "a", "an", "is", "are", "was", "were", "above", "below", "over", "under", "between", "within",
               "near", "around", "about", "but", "while", "whose", "which", "that", "edges", "or", "top", "please",
               "samples", "materials", "records", "cells", "devices", "emitters", "compounds", "values", "value",
               "recorded", "included", "inclusive", "exclusive", "or less", "or more", "and", "x", "times"}
_STOP_WORDS_RE = "|".join(sorted((w for w in _NOT_A_UNIT if " " not in w), key=len, reverse=True))
#: What "per" names that is no unit's denominator: a source or method in capitals ("per MP", "per
#: DFT"), a grouping or counted entity ("per dimensionality", "per device", "per report");
#: "per banana" stays a unit to be refused.
_PER_NOT_A_UNIT = (r"(?:the|a|an|each|every|our|your|this|that|these|those|" + CHAT_SOURCE_PATTERN + r"|"
                   r"(?:DFT|PBE\w*|HSE\w*|GGA\w*|LDA|SCAN|r2SCAN|B3LYP)(?![A-Za-z])|experiments?|measurements?|"
                   r"calculations?|theory|data|values?|charts?|standards?|cations?|anions?|formula(?:s|e)?|materials?|records?|"
                   r"experimental|computed|calculated|measured|reported|theoretical|published|literature|"
                   r"devices?|cells?|reports?|sources?|samples?|compounds?|entries|entry|years?|dimensionalit(?:y|ies)|"
                   r"databases?|collections?|stud(?:y|ies)|papers?|publications?|methods?|functionals?|groups?|"
                   r"categor(?:y|ies)|classes|class|types?|families|family|structures?|phases?)\b")
#: A unit factor written apart from the unit before it ("mA cm-2", "eV kg^-1", "eV / atom").
_FACTOR = r"(?:cm|mm|km|nm|µm|um|m|K|s|ms|g|kg|mg|V|A|W|J|mol|sr|Hz|Pa|eV|L|C|F|N|T|Ω)"
#: A unit's exponent written on it ("cm-2", "cm^-1", "cm⁻²"), one to four, never a year or a count
#: ("% - 2019", "eV -1 (PBE)", "mA/cm2 - 1 sun"), and the next factors of a unit written apart:
#: "S cm^-1", "mA cm-2", "mA·cm⁻²", "W m-1 K-1", "eV per atom", "microvolts per kelvin".
_EXPONENT = r"(?:\^?[-−⁻][1-4](?![\d.,])|\^[1-4](?![\d.,])|[⁺⁻−]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)"
_UNIT = (rf"(?:{_KNOWN_UNIT}|(?!(?:{_STOP_WORDS_RE})(?![A-Za-z0-9/]))"
         rf"[A-Za-zµμΩ][A-Za-z0-9µμΩ/·^²³_-]{{0,12}}(?![A-Za-z0-9/_]))"
         rf"(?:(?<=[A-Za-zµμ]){_EXPONENT})?"
         rf"(?:\s*[·.]?\s*{_FACTOR}(?:{_EXPONENT}|\s*[-−]\s*[1-4](?![\d.,])|\^\s*[-−]?\s*\d{{1,2}}(?![\d.,]))"
         rf"|\s*/\s*(?:{_FACTOR}|atoms?|formula\s+units?)(?![A-Za-z])"
         rf"|\s+per\s+(?:formula\s+units?|(?!{_PER_NOT_A_UNIT})[A-Za-zµμÅ][A-Za-z0-9µμ^·/⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻−-]*))*")

_OR_EQUAL = r"\s+(?:than\s+)?or\s+equal\s+to"
_APPROX_WORD = r"(?:around|about|approximately|approx\.?|ca\.?|roughly|near(?:ly)?|close\s+to|~|≈|약)"
_LO_STRICT = (r"(?:>(?!\s*=)|above|over|beyond|in\s+excess\s+of|(?:more|greater|higher|larger|bigger)\s+than|"
              r"exceed(?:s|ed|ing)?)")
_LO_INCL = (rf"(?:>=|=>|>\s+=|≥|⩾|at\s+least|minimum\s+of|(?:greater|more|higher|larger|bigger)(?:\s+than)?{_OR_EQUAL}|"
            r"equal\s+to\s+or\s+(?:(?:greater|more|higher|larger)\s+than|above|over))")
_HI_STRICT = r"(?:<(?!\s*=)|below|under|beneath|(?:less|lower|smaller|fewer)\s+than)"
_HI_INCL = (rf"(?:<=|=<|<\s+=|≤|⩽|at\s+most|up\s+to|maximum\s+of|(?:less|lower|smaller)(?:\s+than)?{_OR_EQUAL}|"
            r"equal\s+to\s+or\s+(?:(?:less|lower|smaller)\s+than|below|under))")
#: A negation right before a comparator reverses it: "not exceeding 1.5" is <= 1.5, "no lower
#: than 1.1" is >= 1.1. "never" also means every recorded value.
_NEGATION_RE = re.compile(r"(?:\bnot|\bno|\bnever|n['’]t|\bcannot)\s+(?:(?:to|be|been|go|goes|going|lie|lies|fall|falls|"
                          r"rise|rises|get|gets|reach|reaches)\s+)?$", re.I)
_POST = (
    (r"or\s+(?:less|lower|below|fewer|smaller|under)|and\s+(?:below|under|lower|less)(?!\s+than|-)|이하|"
     r"(?:을|를)?\s*넘지\s*(?:않|못)|(?:을|를)?\s*넘(?:으|어)?(?:서는|면)\s*안", "hi", True),
    (r"미만|보다\s*(?:더\s*)?(?:작은|낮은)", "hi", False),
    (r"or\s+(?:more|higher|above|greater|larger|over)|and\s+(?:above|up|higher|over|more)(?!\s+(?:than|to)\b|-)|이상", "lo", True),
    (r"초과|(?:을|를)?\s*넘(?:는|게|은|어|으면)?|보다\s*(?:더\s*)?(?:큰|높은)", "lo", False),
)


def _comparator(word: str, before: str) -> Tuple[str, bool, bool]:
    """Return (side, inclusive, never) of a comparator word; a preceding negation reverses it."""
    lower = bool(re.fullmatch(_LO_INCL + "|" + _LO_STRICT, word, re.I))
    inclusive = bool(re.fullmatch(_LO_INCL + "|" + _HI_INCL, word, re.I))
    neg = _NEGATION_RE.search(before)
    if neg:
        lower, inclusive = not lower, not inclusive
    return ("lo" if lower else "hi"), inclusive, bool(neg and "never" in neg.group(0).lower())

_SENTENCE_END = re.compile(r"(?<!\d)(?<!approx)(?<!\bca)(?<!\bvs)(?<!\be\.g)(?<!\bi\.e)\.|"
                           r"(?<!approx)(?<!\bca)(?<!\bvs)\.(?!\d)|[?!;\n]|"
                           r"(?<=[요다])(?<!보다)(?<!둘\s다)(?<!셋\s다)\s")
_FORMULA_RE = re.compile(r"(?<![A-Za-z0-9])((?:[A-Z][a-z]?(?:\d+(?:\.\d+)?)?){2,})(?![a-z0-9])")

#: Anaphora that refer to a population ("them", "those" and their Korean equivalents): the
#: conditions carry over.
_ANAPHORA_RE = re.compile(
    r"\b(?:them|those|these|that\s+(?:list|top\s*\d*|set)|the\s+rest|among\s+them|of\s+those)\b"
    r"|그중|그\s*중|이\s*중|이들|거기서", re.I)
_CONTINUATION_RE = re.compile(r"^\s*(?:now|redo|again|same|and\s+now|then|그럼|다시|이번엔|이제)\b", re.I)
#: Explicit releases of an earlier condition ("drop the source filter", "any temperature").
_RELEASE_SOURCE_RE = re.compile(r"\b(?:drop|remove|ignore|without)\s+(?:the\s+)?(?:source|collection|database)\s+"
                                r"(?:filter|restriction|limit)|\b(?:any|all|every)\s+(?:source|collection)s?\b|"
                                r"regardless\s+of\s+(?:the\s+)?(?:source|collection)|출처\s*(?:상관\s*없이|무관|제한\s*없이)|"
                                r"전체\s*(?:DB|데이터베이스|코퍼스)", re.I)
_RELEASE_TEMPERATURE_RE = re.compile(r"\b(?:any|all)\s+temperatures?\b|\b(?:drop|remove|ignore)\s+(?:the\s+)?"
                                     r"temperature\s+(?:filter|bound|limit)|온도\s*(?:상관\s*없이|무관)", re.I)
#: "Include DFT values too", "computed values are fine" and their Korean equivalents release
#: measured-only.
_RELEASE_MEASURED_RE = re.compile(
    r"\b(?:include|including|with|allow|also|plus)\s+(?:the\s+)?(?:dft|computed|calculated|theoretical|predicted)\b|"
    r"\b(?:dft|computed|calculated|theoretical)\s+(?:values?|ones?|data|gaps?|results?)\s+(?:are\s+)?(?:fine|ok|okay|"
    r"allowed|too|as\s+well|included)|\b(?:show|give|list)\s+(?:me\s+)?the\s+(?:computed|calculated|dft)\s+ones\b|"
    r"\b(?:any|all)\s+(?:evidence|methods?|study\s+types?)\b|\b(?:dft|computed|calculated)\s+(?:values?\s+)?allowed\b|"
    r"계산\s*값\s*도|계산\s*값\s*포함", re.I)
def _releases_measured(text: str) -> bool:
    """Whether the text releases measured-only ("include DFT values too"; not "DFT excluded")."""
    return any(not _excluding(text, m.start(), m.end()) and not _EXCLUDING_AFTER_RE.match(text[m.end():m.end() + 30])
               for m in _RELEASE_MEASURED_RE.finditer(text))


_RELEASE_PROPERTY_RE = re.compile(r"\b(?:drop|remove|ignore)\s+(?:the\s+)?(?P<what>[\w\s-]{2,30}?)\s+"
                                  r"(?:filter|window|condition|limit)\b", re.I)
#: The same release in Korean word order ("without the efficiency condition", "drop the PCE limit").
_RELEASE_PROPERTY_KO_RE = re.compile(
    r"(?P<what>[A-Za-z가-힣][\w-]{0,15}?)\s*(?:(?:조건|제한|필터|기준|범위)\s*(?:없이|빼고|빼서|제외하고|해제하고|풀고|말고)|"
    r"(?:은|는)?\s*상관\s*없이)", re.I)


#: A word boundary for Latin tokens that also holds against Hangul (a Latin quantity followed by a
#: Korean particle): Python's \b sees no boundary between word characters of different scripts.
_ASCII_B = r"(?:(?<![^\W가-힣ㄱ-ㅎㅏ-ㅣ])(?=[^\W가-힣ㄱ-ㅎㅏ-ㅣ])|(?<=[^\W가-힣ㄱ-ㅎㅏ-ㅣ])(?![^\W가-힣ㄱ-ㅎㅏ-ㅣ]))"
_COMPILED: Dict[Tuple[str, int], "re.Pattern"] = {}


def _ascii(pattern: str, flags: int = re.I) -> "re.Pattern":
    key = (pattern, flags)
    if key not in _COMPILED:
        _COMPILED[key] = re.compile(pattern.replace(r"\b", _ASCII_B), flags)
    return _COMPILED[key]


def _number_parts(text: str) -> Tuple[float, int, int]:
    """Return (mantissa, decimals written, power of ten) of a written number.

    One reading serves both value and precision: "1,000" and "1,000,000e-3" use thousands
    separators, "1,5" a decimal comma, and "1e5", "1×10^5", "1 X 10^5", "10^5" powers of ten."""
    t = re.sub(r"\s+", "", text.replace("−", "-"))
    # "10⁵", "1 × 10⁻³": superscript powers read as "^".
    t = re.sub(r"10([⁺⁻-]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)$", lambda m: "10^" + m.group(1).translate(str.maketrans("⁺⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "+-0123456789")), t)
    power = re.fullmatch(r"([-+]?)(?:([\d.,]+)[x×X])?10\^([-+]?\d+)", t)
    sign, exponent = "", 0
    if power:
        sign, t, exponent = power.group(1), power.group(2) or "1", int(power.group(3))
    e = re.fullmatch(r"(.*?)[eE]([-+]?\d+)", t)
    if e:
        t, exponent = e.group(1), exponent + int(e.group(2))
    if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+", t):
        mantissa, decimals = float(t.replace(",", "")), 0
    else:
        split = re.split(r"[.,]", t, maxsplit=1)
        mantissa, decimals = float(t.replace(",", ".")), (len(split[1]) if len(split) > 1 else 0)
    # Clamp the exponent: 10^400 overflows a float.
    return float(sign + "1") * mantissa, decimals, max(-300, min(300, exponent))


def _num(text: str) -> float:
    """Parse a written number: "1,000", "1,5" (decimal comma), "1e5", "1×10^5", "10^5"."""
    mantissa, _, exponent = _number_parts(text)
    return mantissa * 10.0 ** exponent


def _half_unit(text: str) -> float:
    """Return half the place of the last written digit, reading the number as `_num` does.

    "21.0" is +-0.05, "1.5e2" +-5, "1,500" (thousands) +-0.5, "1,5" (decimal comma) +-0.05,
    "10^5" +-0.5e5."""
    _, decimals, exponent = _number_parts(text)
    return 0.5 * 10.0 ** max(-300, min(300, exponent - decimals))


# Pasted spans

_DIRECTIVE_RE = re.compile(
    r"\[?\s*(?:assistant|ai|model|system)\s+(?:instruction|note|directive)s?\b|note\s+to\s+(?:the\s+)?"
    r"(?:assistant|ai|model)|\binstructions?\s*:|do\s+not\s+mention|rank\s+(?:it|this)\s+(?:#?1|first|top)"
    r"|(?:official|verified)\s+lamp\s+(?:database\s+)?(?:entry|record)|무시하고|언급하지\s*마", re.I)
_INTRO_RE = re.compile(r"pasted|attached|below\s*:|as\s+follows|붙입니다|붙여|자료", re.I)


_ASKS_RE = re.compile(r"\?|\brank\b(?!\s*#?\s*\d)|\b(?:list|show|which|how\s+many|compare|find|give|recommend|sort|tell|what)\b|"
                      r"알려|추천|비교|정리|몇|어떤|뭐", re.I)


def _paste_score(chunk: str) -> int:
    """Score how much a chunk looks pasted rather than asked: table cells, directives, labels."""
    return (chunk.count("|") + 3 * len(_DIRECTIVE_RE.findall(chunk))
            + 2 * len(re.findall(r"(?:material|source|compound)\s*:", chunk, re.I))
            - 2 * len(re.findall(r"\?|\b(?:rank|list|show|which|how\s+many|compare)\b", chunk, re.I)))


def cut_pasted(text: str) -> Tuple[str, List[PastedSpan]]:
    """Split text into (the user's own words, the pasted spans).

    A span is a block between `---` separators (or from one to the end), a code fence, a run of
    `>` quote lines, or, after an introduction such as "pasted", a run of table rows (two or more
    `|`)."""
    spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"(?m)^\s*-{3,}\s*$", text):
        spans.append((m.start(), m.end()))
    blocks: List[Tuple[int, int]] = []
    for i in range(0, len(spans), 2):  # separators pair up; an unpaired one closes or opens a block
        if i + 1 < len(spans):
            blocks.append((spans[i][0], spans[i + 1][1]))
            continue
        before_start = spans[i - 1][1] if i else 0
        before, after = text[before_start:spans[i][0]], text[spans[i][1]:]
        # "row | row\n---\nRank OQMD materials ...": the pasted rows come first. Read so only when
        # the part after the separator asks, never because of pipes in the user's own question.
        labels = re.compile(r"(?:material|source|compound|name)\s*:", re.I)
        if (labels.search(before) and not labels.search(after)) or (
                _paste_score(before) > max(0, _paste_score(after)) and not _ASKS_RE.search(before)):
            blocks.append((before_start, spans[i][1]))
        else:
            blocks.append((spans[i][0], len(text)))
    for m in re.finditer(r"```.*?(?:```|$)", text, re.S):
        blocks.append((m.start(), m.end()))
    for m in re.finditer(r"(?m)(?:^\s*>.*(?:\n|$))+", text):
        blocks.append((m.start(), m.end()))
    if _INTRO_RE.search(text):
        for m in re.finditer(r"(?m)(?:^[^\n|]*\|[^\n|]*\|.*(?:\n|$))+", text):
            blocks.append((m.start(), m.end()))
    blocks = _merge_spans(blocks)
    pasted, own, last = [], [], 0
    for start, end in blocks:
        chunk = text[start:end]
        body = re.sub(r"(?m)^\s*(?:-{3,}|```\w*|>)\s?", "", chunk).strip()
        if not body:
            continue
        pasted.append(PastedSpan(body, tuple(m.group(0) for m in _DIRECTIVE_RE.finditer(body))))
        own.append(text[last:start])
        last = end
    own.append(text[last:])
    return "".join(own).strip(), pasted


def _merge_spans(spans: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


# Mentions

@dataclass(frozen=True)
class _Mention:
    start: int
    end: int
    target: Tuple[str, ...]
    device: bool = False
    computed: bool = False

    @property
    def temperature_quantity(self) -> bool:
        return bool(set(self.target) & TEMPERATURE_QUANTITIES)


#: Quantities named with a stored quantity's word that are another quantity LAMP does not store
#: ("trap density", "density of states", "vacancy formation energy" and their Korean equivalents):
#: read whole, to be refused, never as density or formation energy and never dropped.
_UNSTORED_COMPOUND_RE = re.compile(
    r"\b(?:trap|carrier|charge|defect|dislocation|optical|exciton|energy|power|photocurrent|electron|hole|doping|"
    r"vacancy|interstitial|spin|packing|current\s+noise|(?:exchange|critical|dark|saturation|leakage|recombination|"
    r"limiting|reverse|tunnel(?:l?ing)?)\s+current)\s+densit(?:y|ies)\b|\bdensit(?:y|ies)\s+of\s+states\b|"
    r"\b(?:defect|vacancy|interstitial|antisite|surface|grain[-\s]boundary)\s+formation\s+energ(?:y|ies)\b|"
    r"(?:트랩|캐리어|전하|결함|전위|광학|여기자|에너지|전력|광전류|전자|정공|도핑|공공|상태|암\s*전류|임계\s*전류|"
    r"포화\s*전류|누설\s*전류|교환\s*전류)\s*밀도|"
    r"(?:결함|공공|점\s*결함|표면)\s*형성\s*에너지", re.I)


def _mentions(text: str, property_patterns: Sequence[Tuple["re.Pattern", str]]) -> List[_Mention]:
    """Return every quantity the text names, longest match first, overlaps dropped."""
    found: List[_Mention] = [
        _Mention(m.start(), m.end(), ("unstored:" + re.sub(r"\s+", " ", m.group(0).strip().lower()),))
        for m in _UNSTORED_COMPOUND_RE.finditer(text)]
    for pattern, metric in DEVICE_METRICS:
        for m in _ascii(pattern).finditer(text):
            found.append(_Mention(m.start(), m.end(), (metric,), device=True))
    for pattern, names in SEMANTIC_ALTERNATIVES:
        for m in _ascii(pattern).finditer(text):
            found.append(_Mention(m.start(), m.end(), names))
    for m in _TEMPERATURE_QUANTITY_RE.finditer(text):
        found.append(_Mention(m.start(), m.end(), ("transition_temperature",)))
    for m in _COMPUTED_QUANTITY_RE.finditer(text):
        found.append(_Mention(m.start(), m.end(), (m.group(0).lower(),), computed=True))
    for pattern, name in property_patterns:
        for m in _ascii(pattern.pattern, pattern.flags).finditer(text):
            found.append(_Mention(m.start(), m.end(), (name,)))
    found.sort(key=lambda x: (x.start, -(x.end - x.start)))
    out: List[_Mention] = []
    for mention in found:
        if out and mention.start < out[-1].end:
            continue
        out.append(mention)
    return out


# The reader

def _sentence_bounds(text: str) -> List[Tuple[int, int]]:
    bounds, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        if m.end() - start > 1:
            bounds.append((start, m.end()))
        start = m.end()
    if start < len(text):
        bounds.append((start, len(text)))
    return bounds or [(0, len(text))]


def _sentence_of(bounds, pos):
    for start, end in bounds:
        if start <= pos < end:
            return start, end
    return bounds[-1]


def _window_after(text: str, mention: _Mention, limit: int) -> Optional[Tuple[Condition, int]]:
    """Read a window stated right after a quantity: a comparator with a number, or a range.

    Returns (condition, end offset relative to `mention.end`), or None."""
    tail = text[mention.end:limit]
    # "band gap of Cs2AgBiBr6 above 2 eV", "of 2D materials above 2 eV": a material named between
    # the quantity and its bound, the bound right after it. Not "of CsPbBr3 nanocrystals larger
    # than 10 nm", whose bound is another quantity's.
    lead = re.match(rf"\s*(?:of|for|in)\s+(?:the\s+)?(?:[A-Z][A-Za-z0-9().]*\d[A-Za-z0-9().]*|[0-3]D(?:\s+materials?)?)"
                    rf"(?=\s+(?:(?:is|are|was|were)\s+)?(?:{_LO_INCL}|{_HI_INCL}|{_LO_STRICT}|{_HI_STRICT}|between|within|"
                    rf"from|exactly|{_APPROX_WORD})|\s*[<>≤≥])", tail, re.I)
    if lead:
        shifted = _window_after(text, replace(mention, end=mention.end + lead.end()), limit)
        return (shifted[0], shifted[1] + lead.end()) if shifted else None
    filler = r"[^\d.?!;\n<>≤≥⩾⩽=±]{0,32}?"
    unit = rf"\s*(?P<unit>{_UNIT})?"
    m = re.match(rf"{filler}within\s+(?P<d>{_NUM})\s*(?P<du>{_UNIT})?\s*of\s+(?P<v>{_NUM}){unit}", tail, re.I)
    if m:  # "within 0.20 eV of 1.55 eV": the window centred on the value, edges included
        d, v = _tolerance(abs(_num(m["d"])), m["du"], m["unit"]), _num(m["v"])
        if d is None:
            # "within 0.2 eV/banana of 1.5 eV", "within 0.1 eV of 520 nm": a tolerance with no
            # linear conversion to the centre's unit is refused, never applied in either unit.
            return Condition("approximate", target=mention.target, lo=Bound(v, True), unit=_unit_word(m["unit"] or ""),
                             qualitative=f"within {m['d']} {m['du'] or ''}".strip()), m.end()
        return _window(mention, Bound(round(v - d, 10), True), Bound(round(v + d, 10), True), m["unit"] or m["du"],
                       m.end()), m.end()
    m = re.match(rf"{filler}(?:(?:of|is)\s+|=\s*)?(?P<v>{_NUM})\s*(?P<vu>{_UNIT})?\s*(?:±|\+/-|\+-)\s*"
                 rf"(?P<d>{_NUM})\s*(?P<du>{_UNIT})?", tail, re.I)
    if m:  # "1.5 ± 0.1 eV": the stated window
        centre_unit = m["vu"] or m["du"]
        d = _tolerance(abs(_num(m["d"])), m["du"], centre_unit)
        v = _num(m["v"])
        if d is None:
            return Condition("approximate", target=mention.target, lo=Bound(v, True), unit=centre_unit or "",
                             qualitative=f"± {m['d']} {m['du'] or ''}".strip()), m.end()
        return _window(mention, Bound(round(v - d, 10), True), Bound(round(v + d, 10), True), centre_unit,
                       m.end()), m.end()
    m = re.match(rf"{filler}(?<!or\s)(?<!than\s)(?:of\s+|is\s+|=\s*)?(?:exactly|precisely|equal\s+to|=)\s*"
                 rf"(?P<v>{_NUM}){unit}", tail, re.I)
    if m:  # "exactly 1.5 eV"
        b = Bound(_num(m["v"]), True)
        return _window(mention, b, b, m["unit"], m.end()), m.end()
    m = (re.match(rf"(?P<f>{filler})(?:of\s+|is\s+)?{_APPROX_WORD}\s*(?P<v>{_NUM}){unit}", tail, re.I)
         or re.match(rf"(?P<f>{filler})(?P<v>{_NUM}){unit}\s*(?:정도|가량|내외|쯤|안팎)", tail, re.I))
    if m and not re.search(rf"(?<![a-z])(?:{_LO_INCL}|{_HI_INCL}|{_LO_STRICT}|{_HI_STRICT})", m["f"], re.I) \
            and not _starts_a_range(tail[m.end("v"):]):
        # "around 1.5 eV": the declared tolerance of that quantity, or an explicit no-window
        # condition; "above ~1.5 eV" is a bound.
        return _approximate(mention, _num(m["v"]), m["unit"]), m.end()
    for pattern in (rf"{filler}between\s+(?P<a>{_NUM})\s*(?P<ua>{_UNIT})?\s*and\s+(?P<b>{_NUM}){unit}",
                    rf"{filler}(?:from\s+)?(?P<a>{_NUM})\s*(?P<ua>{_UNIT})?\s*(?P<to>-|–|~|to|(?:down|up)\s+to)\s*"
                    rf"(?P<b>{_NUM}){unit}",
                    rf"{filler}(?P<a>{_NUM})\s*(?P<ua>{_UNIT})?\s*(?:에서|부터)\s*(?P<b>{_NUM}){unit}\s*(?:사이|까지)"):
        m = re.match(pattern, tail, re.I)
        if m:
            if "to" in m.groupdict() and m.group("to") and m.group("to").lower().startswith("down") \
                    and _num(m["a"]) > _num(m["b"]):
                m_a, m_b = m["b"], m["a"]  # "from 3 eV down to 2 eV"
                first = _window(mention, Bound(_num(m_a), True), None, m["unit"] or m["ua"], 0)
                second = _window(mention, None, Bound(_num(m_b), True), m["ua"] or m["unit"], 0)
                return (replace(first, hi=second.hi) if first.unit == second.unit else replace(first, extra=(second,))), m.end()
            first = _window(mention, Bound(_num(m["a"]), True), None, m["ua"] or m["unit"], 0)
            second = _window(mention, None, Bound(_num(m["b"]), True), m["unit"] or m["ua"], 0)
            if first.unit == second.unit:
                return replace(first, hi=second.hi), m.end()
            # "between 1 eV and 2000 meV": each endpoint in its own unit; intersected once converted
            # to the stored unit.
            return replace(first, extra=(second,)), m.end()
    for words in (_LO_INCL, _HI_INCL, _LO_STRICT, _HI_STRICT):
        m = re.match(rf"{filler}(?<![a-z])(?P<w>{words})\s*(?:{_APPROX_WORD}\s*)?(?P<a>{_NUM}){unit}", tail, re.I)
        if m:
            post = re.match(r"\s*(?:or\s+(?:less|more|lower|higher|above|below))", tail[m.end():], re.I)
            if post:  # "above 1 or more" -- contradictory; read the trailing words instead
                break
            side, inclusive, never = _comparator(m["w"], tail[:m.start("w")])
            bound = Bound(_num(m["a"]), inclusive)
            window = _window(mention, bound if side == "lo" else None, bound if side == "hi" else None,
                             m["unit"], m.end())
            return (replace(window, quantifier="every") if never else window), m.end()
    m = re.match(rf"{filler}(?P<a>{_NUM}){unit}\s*(?P<post>[^\d.?!;\n]{{0,20}})", tail, re.I)
    if m:
        for words, side, inclusive in _POST:
            p = re.match(rf"\s*(?:{words})", m["post"], re.I)
            if p:
                bound = Bound(_num(m["a"]), inclusive)
                end = m.start("post") + p.end()
                return _window(mention, bound if side == "lo" else None, bound if side == "hi" else None,
                               m["unit"], end), end
    return None


#: What turns a number into the first end of a range: "around 1.3-1.5 eV" is the range 1.3-1.5 eV,
#: not "around 1.3" with "-1.5" left over.
_RANGE_TAIL = rf"\s*(?P<u1>{_UNIT}\s*)?(?:-|–|~|to)\s*{_NUM}\s*(?P<u2>{_UNIT})?"


def _starts_a_range(tail: str) -> bool:
    """Whether `tail` continues a number into a range in one unit ("-1.5 eV").

    "eV to 30% efficiency" is no range: its two units differ."""
    m = re.match(_RANGE_TAIL, tail, re.I)
    return bool(m) and not (m["u1"] and m["u2"] and _unit_word(m["u1"].strip()) != _unit_word(m["u2"].strip()))


def _unit_fits(unit: str, target: Tuple[str, ...]) -> bool:
    """Whether a value in `unit` can be the quantity's: its stored unit or one convertible to it.

    A value with no unit fits any quantity."""
    from .rag import DEVICE_UNITS, STORED_UNITS, convert_window
    if not unit:
        return True
    stored = next((STORED_UNITS.get(t, DEVICE_UNITS.get(t)) for t in target if t in STORED_UNITS or t in DEVICE_UNITS),
                  None)
    return stored is not None and convert_window(Bound(1.0, True), None, unit, stored) is not None


def _tolerance(d: float, d_unit: Optional[str], centre_unit: Optional[str]) -> Optional[float]:
    """Express a "within d of v" tolerance in the centre's unit: "200 meV of 1.5 eV" is 0.2 eV.

    A difference converts by the scale of a non-reversing conversion (a °C step is a K step); None
    when no such conversion is declared (nm <-> eV is not a scale)."""
    du, cu = (d_unit or "").replace(" ", ""), (centre_unit or "").replace(" ", "")
    if du.lower() in _NOT_A_UNIT:
        du = ""
    if cu.lower() in _NOT_A_UNIT:
        cu = ""
    if not du or not cu:
        return d
    from .rag import _UNIT_CONVERSIONS, _unit_of
    du, cu = _unit_of(du), _unit_of(cu)
    if du == cu:
        return d
    rule = _UNIT_CONVERSIONS.get((du, cu))
    if rule is None or rule[1]:
        return None
    return round(rule[0](d) - rule[0](0.0), 10)


#: The tolerance an approximate value carries where the platform declares one (the criteria parser's
#: own centre rule: band gap and ZT, ±0.1); any other quantity has none.
APPROXIMATE_TOLERANCE = {"bandgap": (0.1, "eV"), "ZT": (0.1, "")}


def _approximate(mention, value: float, unit) -> Condition:
    """Return the window for "<quantity> around <value>", edges included.

    With no declared tolerance for the quantity, return an 'approximate' condition, which every
    consumer refuses by name."""
    declared = APPROXIMATE_TOLERANCE.get(mention.target[0]) if len(mention.target) == 1 else None
    # The tolerance in the unit the value is written in: ±0.1 eV is ±100 meV.
    d = _tolerance(declared[0], declared[1], _unit_word(unit or "")) if declared else None
    if d is None:
        return Condition("approximate", target=mention.target, lo=Bound(value, True), unit=_unit_word(unit or ""))
    w = _window(mention, Bound(round(max(0.0, value - d) if mention.target[0] == "bandgap" else value - d, 10), True),
                Bound(round(value + d, 10), True), unit, 0)
    return replace(w, qualitative=f"approximately {value} (± {d}, the declared tolerance)")


_NEGATED_EXISTENCE_RE = re.compile(
    r"(?:\bno|\bnone\s+of\s+(?:whose|its|their|the)|\bwithout(?:\s+(?:any|a|an))?|\black(?:s|ing)?(?:\s+(?:any|a|an))?|"
    r"\b(?:excluding|exclude|except|other\s+than|(?:do|does)\s*n[o']?t\s+include|leave\s+out|leaving\s+out)\s+"
    r"(?:the\s+)?(?:those|materials|ones|records|compounds|any|entries)?\s*(?:with|having|that\s+have)\s*(?:a|an|any)?|"
    r"\b(?:do|does)\s+not\s+have\s+(?:a|an|any))\s+(?:recorded\s+|reported\s+|measured\s+)?$", re.I)


def _complement(window: Condition) -> Optional[Condition]:
    """Return the complement of a one-sided window (> x -> <= x); None for any other window."""
    if window.extra or (window.lo is None) == (window.hi is None):
        return None
    if window.lo is not None:
        return replace(window, lo=None, hi=Bound(window.lo.value, not window.lo.inclusive))
    return replace(window, hi=None, lo=Bound(window.hi.value, not window.hi.inclusive))


class _Groups:
    """A match's groups with some replaced (two alternatives of one pattern read as one)."""
    def __init__(self, m, **override):
        self._m, self._o = m, override

    def __getitem__(self, name):
        return self._o[name] if name in self._o else self._m[name]

    def start(self, *a):
        return self._m.start(*a)


def _window_before(text: str, mention: _Mention, floor: int) -> Optional[Tuple[Condition, int]]:
    """Read a window stated right before the quantity: "exceed 25% PCE", "at least 90% PLQY"."""
    head = text[max(floor, mention.start - 40):mention.start]
    # "a 1.3-1.5 eV band gap", "a ~1.5 eV band gap": the value as an adjective, with its unit.
    m = re.search(rf"(?:{_APPROX_WORD}\s*)?(?:between\s+(?P<a>{_NUM})\s*(?P<ua>{_UNIT})?\s*and|(?P<a2>{_NUM})\s*"
                  rf"(?P<ua2>{_UNIT})?\s*(?:-|–|~|to))\s*(?P<b>{_NUM})\s*(?P<unit>{_UNIT})[\s-]*$", head, re.I)
    if m and m["a"] is None:
        m = _Groups(m, a=m["a2"], ua=m["ua2"])
    # Two values ("the 1.5 eV and 1.7 eV band gaps") or two units ("85 °C and 1.6 eV") are no range.
    if m and _unit_like(m["unit"]) and (not m["ua"] or _unit_word(m["ua"]) == _unit_word(m["unit"])) \
            and not _is_temperature_unit(m["unit"]):
        # A "300-400 K ZT" is the measurement temperature, read with the temperatures.
        first = _window(mention, Bound(_num(m["a"]), True), None, m["ua"] or m["unit"], 0)
        second = _window(mention, None, Bound(_num(m["b"]), True), m["unit"], 0)
        window = replace(first, hi=second.hi) if first.unit == second.unit else replace(first, extra=(second,))
        return window, mention.start - (len(head) - m.start())
    m = (re.search(rf"{_APPROX_WORD}\s*(?P<a>{_NUM})\s*(?P<unit>{_UNIT})[\s-]*$", head, re.I)
         or re.search(rf"(?P<a>{_NUM})\s*(?P<unit>{_UNIT})\s*(?:정도|가량|내외|쯤|안팎)(?:의|인)?\s*$", head, re.I))
    if m and _unit_like(m["unit"]) and not _is_temperature_unit(m["unit"]):
        return _approximate(mention, _num(m["a"]), m["unit"]), mention.start - (len(head) - m.start())
    for words in (_LO_INCL, _HI_INCL, _LO_STRICT, _HI_STRICT):
        m = re.search(rf"(?<![a-z])(?P<w>{words})\s*(?P<a>{_NUM})\s*(?P<unit>{_UNIT})?\s*(?:of\s+)?$", head, re.I)
        if m:
            side, inclusive, _ = _comparator(m["w"], head[:m.start("w")])
            bound = Bound(_num(m["a"]), inclusive)
            start = mention.start - (len(head) - m.start())
            return _window(mention, bound if side == "lo" else None, bound if side == "hi" else None,
                           m["unit"], 0), start
    return None


_UNIT_WORDS = ((r"milli-?electron[-\s]?volts?|밀리\s*전자\s*볼트", "meV"), (r"electron[-\s]?volts?|전자\s*볼트", "eV"),
               (r"per\s?cent|퍼센트", "%"), (r"kelvins?|켈빈", "K"), (r"microns?|마이크로미터|마이크론", "µm"),
               (r"나노미터", "nm"), (r"밀리볼트", "mV"), (r"볼트", "V"), (r"옹스트롬", "Å"))
#: A unit's name read as its symbol: "nanometers" is nm, "millivolts" mV, "kilojoules" kJ.
_UNIT_PREFIXES = {"peta": "P", "tera": "T", "giga": "G", "mega": "M", "kilo": "k", "hecto": "h", "deci": "d",
                  "centi": "c", "milli": "m", "micro": "µ", "nano": "n", "pico": "p", "femto": "f", "atto": "a"}
_UNIT_BASES = ((r"met(?:er|re)s?", "m"), (r"grams?", "g"), (r"seconds?|secs?", "s"), (r"amp(?:ere)?s?", "A"),
               (r"kelvins?", "K"), (r"moles?", "mol"), (r"hertz", "Hz"), (r"newtons?", "N"), (r"pascals?", "Pa"),
               (r"joules?", "J"), (r"watts?", "W"), (r"coulombs?", "C"), (r"volts?", "V"), (r"farads?", "F"),
               (r"ohms?", "Ω"), (r"siemens", "S"), (r"teslas?", "T"), (r"electron-?volts?", "eV"),
               (r"lit(?:er|re)s?", "L"), (r"[aå]ngstr[oö]ms?", "Å"), (r"hours?|hrs?", "h"), (r"minutes?|mins?", "min"),
               (r"atoms?", "atom"))
_UNIT_NAME_SPLIT_RE = re.compile(r"(?P<prefix>" + "|".join(_UNIT_PREFIXES) + r")?-?(?P<base>"
                                 + "|".join(f"(?:{b})" for b, _ in _UNIT_BASES) + ")", re.I)


def _unit_word(unit: str) -> str:
    if re.search(r"\s+per\s+", unit or ""):
        # "microvolts per kelvin" is µV/K, "eV per atom" eV/atom; in "eV per MBJ" or "eV per
        # OptB88vdW" the word after "per" is a method label, not a denominator.
        head, *tail = re.split(r"\s+per\s+", unit)
        if tail and len(re.findall(r"[A-Z]", tail[0])) >= 2 and not (
                re.fullmatch(_KNOWN_UNIT, tail[0], re.I) or _SI_SYMBOL_RE.fullmatch(tail[0]) or _UNIT_NAME_RE.fullmatch(tail[0])):
            return _unit_word(head)
        return "/".join(_unit_word(part) for part in [head] + tail)
    for pattern, symbol in _UNIT_WORDS:
        if re.fullmatch(pattern, unit or "", re.I):
            return symbol
    m = _UNIT_NAME_SPLIT_RE.fullmatch(unit or "")
    if m:
        base = next(symbol for pattern, symbol in _UNIT_BASES if re.fullmatch(pattern, m["base"], re.I))
        return (_UNIT_PREFIXES[m["prefix"].lower()] if m["prefix"] else "") + base
    return unit


#: Every element symbol: after a number, "S", "Mg", "Pa" name an element ("ZT above 1 S or Se
#: compounds"), never a unit.
_ELEMENT_SYMBOLS = set(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr "
    "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc "
    "Lv Ts Og".split())
#: A unit written as an SI symbol ("J", "kJ", "Hz", "MPa", "Ry") or a unit's name ("hartree",
#: "bohr"): undeclared ones are kept so they are refused, never replaced by the stored unit. A bare
#: capital letter other than J is a variable or an element ("T=600 K", "S"), never a unit.
_SI_SYMBOL_RE = re.compile(
    r"[kmµμunpfcdh]?(?:mol|cd|Hz|Pa|Wb|lm|lx|Bq|Gy|Sv|eV|bar|cal|Da)|[kmµμunpfcdh][AJWCVFΩSTHLN]|[kmµμunpfcdh]?[gsm]|"
    r"[MGTP](?:Hz|Pa|J|W|eV|bar|Da|g|s|m|Ω)|J|Ry|Ha|Eh|atm|Torr|Å|min|rpm|BTU|h")
_UNIT_NAME_RE = re.compile(
    r"(?:peta|tera|giga|mega|kilo|hecto|deca|deci|centi|milli|micro|nano|pico|femto|atto)?-?"
    r"(?:met(?:er|re)s?|grams?|seconds?|amp(?:ere)?s?|kelvins?|moles?|candelas?|hertz|newtons?|pascals?|joules?|"
    r"watts?|coulombs?|volts?|farads?|ohms?|siemens|webers?|teslas?|henr(?:y|ies)|lumens?|lux|becquerels?|grays?|"
    r"sieverts?|calories?|electron-?volts?|lit(?:er|re)s?|bars?|atmospheres?|torr|[aå]ngstr[oö]ms?|daltons?|"
    r"gauss|ergs?|dynes?|poises?|debyes?|barns?|hartrees?|rydbergs?|bohrs?|microns?|inch(?:es)?|feet|foot|"
    r"wave-?numbers?|btus?|hours?|hrs?|minutes?|mins?)|하트리|리드버그|보어", re.I)
#: Organic A-site cations people abbreviate ("1.5 MA cation", "MA or FA"): never a unit.
_CATIONS = {"MA", "FA", "GA", "EA", "BA", "PEA", "DMA", "BDA", "PA", "AVA", "TEA", "TMA", "IPA", "HA", "OA", "PMA"}
#: Words after a number that read as a symbol and are not one: time of day, English words.
_NOT_A_SYMBOL = {"am", "pm", "as", "at", "in", "is", "us", "me", "my"}


def _unit_like(unit: str) -> bool:
    """Whether a token is a unit: a declared unit, an SI symbol or unit name ("J", "Hz", "joules",
    "Ry"), or a token shaped like one ("eV/banana", "W/m2K", "Ω").

    "only", "too", "each", "MP", element symbols ("S", "Mg") and cations ("MA", "GA") are words,
    not units."""
    if re.search(r"\s+per\s+", unit):
        # "microvolts per kelvin", "eV per atom": every factor a unit; the last may be an entity.
        head, *tail = re.split(r"\s+per\s+", unit)
        # "S per cm": with "per", a symbol is a unit even when it also names an element.
        # With a unit before it, the whole is a unit; an undeclared one ("eV per banana") is kept
        # to be refused.
        return _unit_like(head) or bool(re.fullmatch(r"[A-Za-zµμΩ]{1,3}", head))
    if re.fullmatch(_KNOWN_UNIT, unit, re.I) or re.search(r"[/·^µμΩ\d%°⁻⁰¹²³⁴⁵⁶⁷⁸⁹]", unit) or _UNIT_NAME_RE.fullmatch(unit):
        return True
    if unit in _ELEMENT_SYMBOLS or unit in _CATION_TOKENS or unit in _CATIONS or unit.lower() in _NOT_A_SYMBOL:
        return False
    return bool(_SI_SYMBOL_RE.fullmatch(unit))


def _window(mention, lo, hi, unit, _end) -> Condition:
    # Judged as written ("siemens", "pascals" are units, though their symbols S and Pa are also
    # elements), then read as its symbol.
    written = (unit or "").strip()
    unit = "" if written.lower() in _NOT_A_UNIT or not _unit_like(written) else _unit_word(written).replace(" ", "")
    return Condition("window", target=mention.target, lo=lo, hi=hi, unit=unit)


_CONTINUED_BOUND_RE = re.compile(
    rf"\s*(?:,\s*(?:(?:and|but)\s+(?:also\s+)?)?|;\s*|(?:and|but)\s+(?:also\s+)?)(?P<neg>(?:not|no|never)\s+)?"
    rf"(?P<w>{_LO_INCL}|{_HI_INCL}|{_LO_STRICT}|{_HI_STRICT})\s*(?P<a>{_NUM})\s*(?P<unit>{_UNIT})?", re.I)
#: What follows a number that counts rather than bounds: "at most 10 devices", "at least 10 of
#: them", "more than 3 elements".
_COUNTED_RE = re.compile(
    r"\s*(?:of\s+(?:them|these|those)\b|(?:materials?|devices?|cells?|records?|values?|samples?|compounds?|"
    r"results?|items?|entries|elements?|measurements?|sources?|rows?|emitters?|structures?|data\s+points?|"
    r"candidates?|options?|examples?|atoms?|sites?|species|layers?|papers?|reports?|studies|groups?|phases?|"
    r"more|or\s+more\s+of)\b|(?<=\d)\s*(?:개|건|가지|종)(?![가-힣]))", re.I)
#: A second Korean postfix bound continuing the first (the Korean form of "1.5 eV or more and 2 eV
#: or less").
_CONTINUED_POST_RE = re.compile(
    rf"\s*(?:,|그리고|이고|이며|면서)?\s*(?P<a>{_NUM})\s*(?P<unit>{_UNIT})?\s*(?P<post>이하|미만|이상|초과)", re.I)


def _continued_bound(text: str, mention: "_Mention", start: int,
                     previous: Optional[Condition] = None) -> Optional[Tuple[Condition, int]]:
    """Read a second bound on the same quantity, unnamed ("above 1 eV and below 2000 meV")."""
    m = _CONTINUED_BOUND_RE.match(text, start)
    if m and _COUNTED_RE.match(text, m.end()):
        return None  # ", at most 10 devices": a count
    if m and not re.match(r"\s*,?\s*(?:and|but)\b|\s*;", text[start:m.start("w")], re.I) \
            and not re.fullmatch(_KNOWN_UNIT, m.group("unit") or "", re.I):
        # A comma-only continuation with no unit is this quantity's only when the quantity has none
        # ("ZT above 1, below 2") or it bounds the other side ("above 1 eV, below 2"); "above 2 eV,
        # at least 3" is not.
        side, _, _ = _comparator(m.group("w"), m.group("neg") or "")
        first_side = None if previous is None else ("lo" if previous.lo is not None and previous.hi is None else
                                                    "hi" if previous.hi is not None and previous.lo is None else None)
        if not (previous is not None and (not previous.unit or (first_side and side != first_side))):
            return None
    never = False
    if m:
        side, inclusive, never = _comparator(m.group("w"), m.group("neg") or "")
    else:
        m = _CONTINUED_POST_RE.match(text, start)
        if not m:
            return None
        side = "hi" if m.group("post") in ("이하", "미만") else "lo"
        inclusive = m.group("post") in ("이하", "이상")
    unit = m.group("unit") or ""
    if unit and _COUNTED_RE.fullmatch(" " + unit):
        return None  # "above 1 eV and more than 3 elements": a count, not this quantity's bound
    # Any other unit, an undeclared one included ("below 2 eV/banana"), is kept to be refused where
    # it is converted, never dropped.
    if (not unit or unit.lower() in _NOT_A_UNIT or not _unit_like(unit)) and previous is not None and previous.unit:
        unit = previous.unit  # "above 1 eV, below 2": the same unit
    bound = Bound(_num(m.group("a")), inclusive)
    lower = side == "lo"
    window = _window(mention, bound if lower else None, None if lower else bound, unit, 0)
    # "... and never exceeding 2 eV": the bound holds for every recorded value.
    return (replace(window, quantifier="every") if never else window), m.end()


def _is_temperature_unit(unit: str) -> bool:
    return unit in ("K", "°C", "℃")


_T_UNIT = r"(?:K|°C|℃|kelvins?)(?![A-Za-z]|-(?!ish\b)[A-Za-z])"  # "2 K-based" is potassium
#: Temperature ranges ("between 500 and 700 K", "at 300-400 K", "within 50 K of 600 K") and
#: trailing forms ("600 K or higher"), read before the single bounds.
_T_RANGE_RE = re.compile(
    rf"(?:between|from)\s+(?P<a>{_NUM})\s*(?P<ua>{_T_UNIT})?\s*(?:and|to|-|–)\s*(?P<b>{_NUM})\s*(?P<u>{_T_UNIT})"
    rf"|(?:at|in|over|across)\s+(?:the\s+)?(?:(?:temperature\s+)?range\s+(?:of\s+)?)?(?P<a3>{_NUM})\s*(?P<ua3>{_T_UNIT})?\s*"
    rf"(?:-|–|~|to)\s*(?P<b3>{_NUM})\s*(?P<u3>{_T_UNIT})"
    rf"|(?P<a4>{_NUM})\s*(?P<ua4>{_T_UNIT})?\s*(?:~|-|–|에서)\s*(?P<b4>{_NUM})\s*(?P<u4>{_T_UNIT})\s*(?:사이|에서|까지)"
    rf"|\bT\s*(?P<tw>>=|<=|≥|≤|>|<)\s*(?P<tv>{_NUM})\s*(?P<tu>{_T_UNIT})"
    rf"|within\s+(?P<d>{_NUM})\s*(?:K|°C|℃)\s+of\s+(?P<v>{_NUM})\s*(?P<uv>{_T_UNIT})"
    rf"|(?:at\s+)?(?P<p>{_NUM})\s*(?P<up>{_T_UNIT})\s+(?P<post>or\s+(?:higher|above|more|over|greater|hotter)|and\s+above|"
    rf"or\s+(?:lower|below|less|under|colder)|and\s+below)"
    # a bare range, "a 300-400 K ZT above 1"
    rf"|(?P<a5>{_NUM})\s*(?P<ua5>{_T_UNIT})?\s*(?:-|–|~|to)\s*(?P<b5>{_NUM})\s*(?P<u5>{_T_UNIT})", re.I)


def _kelvin(value: float, unit: Optional[str]) -> float:
    return round(value + 273.15, 2) if unit in ("°C", "℃") else value


def _temperatures(text: str, consumed: List[Tuple[int, int]]) -> List[Condition]:
    out = []
    taken = list(consumed)
    for m in _T_RANGE_RE.finditer(text):
        if any(s <= m.start() < e for s, e in taken):
            continue
        if _TEMPERATURE_QUANTITY_BEFORE_RE.search(text[max(0, m.start() - 30):m.start()]):
            continue  # "Curie temperature between 300 and 400 K" is that quantity's value
        if m["tw"]:
            v, w = _kelvin(_num(m["tv"]), m["tu"]), m["tw"]
            b = Bound(v, "=" in w or w in "≥≤")
            lo, hi = (b, None) if w[0] in ">≥" else (None, b)
        elif m["a"] is not None or m["a3"] is not None or m["a4"] is not None or m["a5"] is not None:
            a, b = ((m["a"], m["b"]) if m["a"] is not None else (m["a3"], m["b3"]) if m["a3"] is not None
                    else (m["a4"], m["b4"]) if m["a4"] is not None else (m["a5"], m["b5"]))
            u = m["u"] or m["u3"] or m["u4"] or m["u5"]
            ua = m["ua"] or m["ua3"] or m["ua4"] or m["ua5"] or u
            lo, hi = Bound(_kelvin(_num(a), ua), True), Bound(_kelvin(_num(b), u), True)
        elif m["d"] is not None:
            v, d = _kelvin(_num(m["v"]), m["uv"]), abs(_num(m["d"]))
            lo, hi = Bound(v - d, True), Bound(v + d, True)
        else:
            v = Bound(_kelvin(_num(m["p"]), m["up"]), True)
            lo, hi = (v, None) if re.search(r"higher|above|more|over|greater|hotter", m["post"], re.I) else (None, v)
        out.append(Condition("temperature", lo=lo, hi=hi, unit="K", span=(m.start(), m.end()),
                             qualitative="bare range" if m["a5"] is not None else ""))
        taken.append((m.start(), m.end()))
    consumed = taken
    pattern = re.compile(
        rf"(?P<w>at\s+or\s+below|at\s+or\s+above|up\s+to|no\s+more\s+than|below|under|above|over|"
        rf"≤|<=|≥|>=|at|near|around|about|approximately|temperature\s*[:=]|T\s*=|\()\s*(?P<a>{_NUM})\s*"
        rf"(?P<u>K|°C|℃|kelvins?)(?![A-Za-z]|-(?!ish\b)[A-Za-z])"
        rf"|(?P<a2>{_NUM})\s*(?P<u2>K|°C|℃)\s*(?P<w2>이하|이상|미만|초과|부근|근처|에서)", re.I)
    numeric = len(out)
    for m in pattern.finditer(text):
        if any(s <= m.start() < e for s, e in consumed):
            continue
        word = (m["w"] or m["w2"] or "").lower()
        value = _num(m["a"] or m["a2"])
        if (m["u"] or m["u2"]) in ("°C", "℃"):
            value = round(value + 273.15, 2)
        # "at 600 K", "(600 K)", "Temperature: 600 K", "T = 600 K": +-25 K, never below 0 K.
        if word in ("at", "near", "around", "about", "approximately", "부근", "근처", "에서", "(") \
                or re.fullmatch(r"temperature\s*[:=]|t\s*=", word):
            lo, hi = Bound(max(0.0, value - 25.0), True), Bound(value + 25.0, True)
        elif word in ("at or below", "up to", "no more than", "≤", "<=", "이하"):
            lo, hi = None, Bound(value, True)
        elif word in ("below", "under", "미만"):
            lo, hi = None, Bound(value, False)
        elif word in ("at or above", "≥", ">=", "이상"):
            lo, hi = Bound(value, True), None
        else:
            lo, hi = Bound(value, False), None
        out.append(Condition("temperature", lo=lo, hi=hi, unit="K", span=(m.start(), m.end())))
    if len(out) == numeric == 0:
        # "at room temperature" or "RT" (or a Korean equivalent) as the only temperature named:
        # 280-320 K, the same reading as the recommender's.
        for m in re.finditer(r"room[-\s]*temp(?:erature)?s?\b|(?-i:(?<![\w.])RT(?![\w.]))|상온|실온", text, re.I):
            if not any(s <= m.start() < e for s, e in consumed):
                out.append(Condition("temperature", lo=Bound(280.0, True), hi=Bound(320.0, True), unit="K",
                                     span=(m.start(), m.end())))
    return out


_SUPERLATIVE_DESC = (r"wid(?:est|er)|highest|largest|maximum|greatest|best|champion|strongest|biggest|"
                     r"deepest|longest|top")
_SUPERLATIVE_ASC = r"narrow(?:est|er)|lowest|smallest|minimum|least|weakest|poorest|worst|shallowest|shortest"
_KO_DESC = r"(?:제일|가장)\s*(?:큰|높은|넓은|긴|깊은|좋은)|최고|최대|상위"
_KO_ASC = r"(?:제일|가장)\s*(?:작은|낮은|좁은|짧은)|최저|최소|하위"
#: A Korean ordering written after its quantity without a "most" word (the Korean forms of "the
#: high-ZT ones, five of them" or "in order of high Voc"). It ranks only the quantity right
#: before it (`_ko_bound_mention`).
_KO_ORDER_TAIL = r"\s*(?:거|것|게|애|순(?:서)?(?:대로|로|으로)?|쪽)(?![가-힣])|\s*(?:거|것)(?=[을를은는이가만들도])"
_KO_POST_DESC = rf"(?:높은|큰|넓은|긴|깊은|좋은)(?:{_KO_ORDER_TAIL})|내림차순"
_KO_POST_ASC = rf"(?:낮은|작은|좁은|짧은|얕은)(?:{_KO_ORDER_TAIL})|오름차순"
#: Particles that may stand between a quantity and the Korean cue that ranks it.
_KO_CUE_GAP = r"\s*(?:값|수치)?\s*(?:이|가|은|는|도|의)?\s*"
_KO_COUNT_AFTER_RE = re.compile(
    r"[^.?!,;:\n]{0,15}?(?<![\d.])(?P<n>\d{1,3}|하나|한|둘|두|셋|세|넷|네|다섯|여섯|일곱|여덟|아홉|열)\s*"
    r"(?:개|건|가지|종|위)(?!\s*(?:이상|이하|미만|초과|넘|이내))")


def _ko_bound_mention(text: str, mentions, s_start: int, cue_start: int):
    """The quantity a Korean cue ranks: the one right before it in its clause, with only particles
    between."""
    before = [x for x in mentions if s_start <= x.start and x.end <= cue_start]
    if before and re.fullmatch(_KO_CUE_GAP, text[before[-1].end:cue_start]):
        return before[-1]
    return None


def _ko_n_after(text: str, end: int) -> Optional[int]:
    """The count written after a Korean ranking cue, never an input population before it."""
    m = _KO_COUNT_AFTER_RE.match(text, end)
    if not m:
        return None
    word = m.group("n")
    return int(word) if word.isdigit() else _NUMBER_WORDS.get(word)


def _n_near(text: str, start: int, end: int) -> Optional[int]:
    around = text[max(0, start - 30):end + 30]
    # Every candidate, not only the first: "the highest band gap, top 3" reads "highest band" first.
    for pattern, where in ((r"\b(?:top|best|first|highest|lowest|largest|smallest)[\s-]+(\d+|[a-z]+)\b", around),
                           (r"\b(\d+|[a-z]+)\s+(?:highest|best|largest|lowest|smallest|top)\b", around),
                           (r"상위\s*(\d+|한|두|세|네|다섯)\s*(?:개|건|가지)?", around),
                           (r"\bthe\s+(\d+|[a-z]+)\s+(?:compounds?|materials?|records?|compositions?|devices?|"
                            r"cells?|samples?|emitters?)\s+(?:whose|with|having|that\s+have)\b", text[max(0, start - 120):end + 10])):
        for m in re.finditer(pattern, where, re.I):
            word = m.group(1).lower()
            n = int(word) if word.isdigit() else _NUMBER_WORDS.get(word)
            if n:
                return n
    return None


def _n_named(sentence: str) -> Optional[int]:
    """"name the three compounds", "give the five records": the N a ranking asks for."""
    m = re.search(r"\b(?:name|give|list|show|find|return)\s+(?:me\s+)?(?:the\s+)?(\d+|[a-z]+)\s+"
                  r"(?:(?:(?-i:[A-Z][A-Za-z0-9.]*)|solar[- ]cell|solar|perovskite|photovoltaic|screened)\s+){0,4}"
                  r"(?:compounds?|materials?|records?|compositions?|devices?|cells?|samples?|emitters?)\b",
                  sentence, re.I)
    if not m:
        return None
    word = m.group(1).lower()
    return int(word) if word.isdigit() else _NUMBER_WORDS.get(word)


#: Verbs before "by" that do not rank: "filtered by", "grouped by", "colored by", "normalized by".
_NOT_RANKING_BY = (r"(?:filter(?:ed)?|group(?:ed)?|accompanied|broken\s+(?:down|out)|split|then|followed|"
                   r"categori[sz]ed|classified|colou?red|normali[sz]ed|separated|divided|weighted|scaled|labell?ed|"
                   r"clustered|binned|stratified|partitioned)")  # "as measured by ZT" ranks


def _ranks(text: str, mentions: List[_Mention], bounds) -> List[Condition]:
    found = []
    pattern = re.compile(
        rf"\b(?P<desc>{_SUPERLATIVE_DESC})\b|\b(?P<asc>{_SUPERLATIVE_ASC})\b|(?P<kdesc>{_KO_DESC})|"
        rf"(?P<kpdesc>{_KO_POST_DESC})|(?P<kpasc>{_KO_POST_ASC})|"
        # "Rank them by", "rank the materials by", "Rank StarryData2 materials by": a population as
        # a plural noun, never "rank each composition by the median ..." (a selection criterion). Ranked
        # by the quantity named right after "by" only ("Rank materials by dimensionality" names none
        # it can rank by).
        rf"(?P<kasc>{_KO_ASC})|\b(?P<by>(?:rank|sort|order)(?:ed)?\s+(?:[^.?!;\n]{{0,60}}?\s)?by)\b"
        rf"(?!\s+(?:the\s+)?(?:highest|lowest|largest|smallest|best|worst|top)\b)",
        re.I)
    previous_end = None
    rejected = (None, None)   # (quantity of a rejected Korean ordering, its sentence start)
    for m in pattern.finditer(text):
        before = text[:m.start()]
        if re.search(r"\bat\s+(?:the\s+)?(?:very\s+)?$", before, re.I):
            continue  # "at least", "at most": a bound, not a superlative
        if not m.group("by") and previous_end is not None and re.fullmatch(r"\s*", text[previous_end:m.start()]):
            previous_end = m.end()
            continue  # "the highest minimum band gap": the second word names the aggregate
        previous_end = m.end()
        s_start, s_end = _sentence_of(bounds, m.start())
        if not m.group("by") and re.search(
                r"(?:\bfor\s+each\b|\beach\s+one(?:'s)?\b|\beach\b|\bper\s+(?:item|material|record|emitter)\b|"
                r"각각)[^.?!;]{0,25}$", text[max(s_start, m.start() - 40):m.start()], re.I):
            continue  # "for each one, the best EQE", "give each one's lowest gap": a per-item value
        direction = "asc" if (m.group("asc") or m.group("kasc") or m.group("kpasc")) else "desc"
        korean = bool(m.group("kdesc") or m.group("kasc") or m.group("kpdesc") or m.group("kpasc"))
        bound_to = _ko_bound_mention(text, mentions, s_start, m.start()) if korean else None
        if korean and re.match(r"\s*(?:거|것|게)?\s*(?:이|가)?\s*(?:말고|아닌|아니라|아니고|빼고|제외)", text[m.end():]):
            # A rejected ordering ("not the high ones", "not low Voc but high"): the next cue in the
            # sentence ranks the same quantity.
            rejected = (bound_to, s_start)
            continue
        if (m.group("kpdesc") or m.group("kpasc")) and bound_to is None:
            if rejected[0] is not None and rejected[1] == s_start:
                bound_to = rejected[0]
            else:
                # No quantity right before the postfix cue. A number compared before it makes the
                # cue a bound ("higher than 1.5 eV"); with none, there is nothing it ranks.
                continue
        if m.group("by"):
            tail = text[m.end():m.end() + 20].lower()
            direction = "asc" if re.match(r"\s*(?:min|lowest|smallest|ascending)", tail) else "desc"
        if bound_to is not None:
            target, aggregation = bound_to, "point"
        else:
            target, aggregation = _rank_target(text, mentions, m.start(), m.end(), s_start, s_end)
        span = (m.start(), m.end())
        if m.group("by"):
            # The quantity named right after "by", up to three qualifiers between: evidence
            # ("measured", "DFT"), direction ("ascending", "how small the"), aggregate ("mean",
            # "max"), each kept. "By the median of its recorded band gaps" (a selection criterion)
            # is no ranking.
            if target is None or target.start < m.end() or not re.fullmatch(
                    r"\s*(?:(?:the|their|its|both)\s+)?(?:(?:max(?:imum)?|min(?:imum)?|peak|average|mean|median|recorded|measured|"
                    r"experimental|reported|computed|calculated|DFT|PBE\w*|HSE\w*|optical|certified|champion|ascending|"
                    r"descending|increasing|decreasing|how\s+(?:large|small|high|low|big)\s+the)[\s'\"]+){0,3}['\"]?",
                    text[m.end():target.start], re.I):
                continue
            span = (m.end() - 2, m.end())  # bound where "by" is, after the population it names
            if re.search(r"\b(?:ascending|increasing|how\s+(?:small|low))\b", text[m.end():target.start], re.I) \
                    or re.match(r"[^.;,]{0,20}?\b(?:in\s+ascending\s+order|ascending|from\s+(?:the\s+)?(?:lowest|smallest))\b",
                                text[target.end:], re.I):
                direction = "asc"
        if target is not None and target.start >= m.end() and re.fullmatch(
                r"\s*(?:(?:recorded|reported|measured)\s+)?", text[m.end():target.start], re.I) and re.match(
                rf"\w{{0,2}}\s*(?:values?\s+)?(?:is|are|was|were|lies?)\s+(?:{_LO_INCL}|{_HI_INCL}|{_LO_STRICT}|"
                rf"{_HI_STRICT}|between|within|not|no)", text[target.end:], re.I):
            continue  # "whose highest band gap is below 2 eV": a window on the aggregate
        n = ((_ko_n_after(text, m.end()) if korean else None)
             or _n_near(text, m.start(), m.end()) or _n_named(text[s_start:s_end]))
        found.append(Condition("rank", target=target.target if target else (), direction=direction,
                               n=n, routed=bool(target and target.computed),
                               aggregation=aggregation, span=span))
    # "Top five please": an N with no superlative word nearby, ranked by the quantity a later "by"
    # names ("Top 5 materials with band gap above 1 eV by ZT").
    for m in re.finditer(r"\btop[\s-]+(\d+|[a-z]+)\b", text, re.I):
        if not any(c.span[0] <= m.start() < c.span[1] + 30 for c in found):
            n = _n_near(text, m.start(), m.end())
            if n:
                s_end = _sentence_of(bounds, m.start())[1]
                by = re.compile(r"\bby\s+(?:(?:the\s+)?(?:highest|largest|max(?:imum)?)\s+)?$", re.I)
                named = [x for x in mentions if m.end() <= x.start < s_end and by.search(text[m.end():x.start])
                         and not re.search(r"\b" + _NOT_RANKING_BY + r"\s+by\s+", text[m.end():x.start], re.I)]
                found.append(Condition("rank", direction="desc", n=n, target=named[0].target if named else (),
                                       routed=bool(named and named[0].computed), span=(m.start(), m.end())))
    return found  # folded per population after binding


def _rank_target(text, mentions, start, end, s_start, s_end):
    """Return (mention, aggregation) for the quantity a superlative ranks.

    That is the next quantity named in its sentence ("the highest PCE", "rank them by max EQE"),
    else the one just before it (Korean word order)."""
    # "Top 5 materials with band gap above 1 eV by ZT": the quantity a later ranking "by" names,
    # when none follows the superlative itself; never "filtered by", "grouped by", "accompanied by".
    after = [x for x in mentions if end <= x.start < min(s_end, end + 60)]
    by_word = r"\bby\s+(?:(?:the\s+)?(?:highest|largest|lowest|smallest|max(?:imum)?|min(?:imum)?|peak)\s+)?$"
    not_ranking = r"\b" + _NOT_RANKING_BY + r"\s+" + by_word
    if not (after and re.fullmatch(r"[\s\d-]*", text[end:after[0].start])):
        by = [x for x in mentions if end <= x.start < s_end and re.search(by_word, text[end:x.start], re.I)
              and not re.search(not_ranking, text[end:x.start], re.I)]
        # "Top 5 materials categorized by band gap": a grouping's quantity is not the ranked one.
        after = by[:1] or [x for x in after if not re.search(not_ranking, text[end:x.start], re.I)]
    if after:
        gap = text[end:after[0].start].lower()
        aggregation = ("peak" if re.search(r"\b(?:peak|max(?:imum)?)\b", gap) else
                       "median" if "median" in gap else
                       "mean" if re.search(r"\b(?:mean|average)\b", gap) else
                       "min" if re.search(r"\b(?:min(?:imum)?|lowest)\b", gap) else "point")
        return after[0], aggregation
    before = [x for x in mentions if s_start <= x.start < start]
    if before:
        gap = text[max(s_start, before[-1].start - 12):before[-1].start].lower()
        aggregation = "peak" if re.search(r"\bpeak\b", gap) else "point"
        return before[-1], aggregation
    return None, "point"


def _fold_ranks(found: List[Condition]) -> List[Condition]:
    """Fold the rankings of one question span into one.

    "whose lowest recorded band gaps are the lowest" is a per-compound minimum ranked ascending:
    the superlative that names the quantity is the aggregation, the one that does not is the
    order."""
    if len(found) < 2:
        return found
    named = [c for c in found if c.target]
    bare = [c for c in found if not c.target]
    if named and bare:
        main = named[0]
        agg = {"asc": "min", "desc": "peak"}[main.direction] if main.aggregation == "point" else main.aggregation
        n = next((c.n for c in found if c.n), None)
        return [replace(main, direction=bare[-1].direction, aggregation=agg, n=n)]
    if len(named) > 1 and len({c.target for c in named}) == 1:
        # "the lowest recorded band gaps are the lowest": the first superlative modifies the
        # quantity (a per-member aggregate), the last one orders.
        first, last = named[0], named[-1]
        agg = first.aggregation if first.aggregation != "point" else {"asc": "min", "desc": "peak"}[first.direction]
        n = next((c.n for c in found if c.n), None)
        return [replace(last, aggregation=agg, n=n)]
    return found


def _quantifier(text: str, start: int, end: int, bounds, clause_start: Optional[int] = None,
                window_end: Optional[int] = None) -> Tuple[str, str]:
    """Return ('every' | 'any', 'composition' | 'record') for the window on the quantity at `start`.

    The quantifier is read from that quantity's own clause, from the previous quantity named to the
    end of its window phrase, so "every band gap between 1 and 2 eV and work function above 6.5 eV"
    puts 'every' on the band gap only."""
    s_start, s_end = _sentence_of(bounds, start)
    lo = s_start if clause_start is None else max(s_start, clause_start)
    near = text[lo:min(s_end, end + 80)].lower()
    after = text[end:min(s_end, end + 80)].lower()
    inside = text[end:window_end].lower() if window_end else ""
    every = bool(_every_before(text[lo:start])
                 or re.search(r"\bevery\b|\ball\s+(?:lie|are|fall)\b|\bvalues?\s+all\b|모든", inside)
                 or re.match(r"^\w{0,2}\s*(?:values?\s+)?(?:recorded\s+(?:on\s+it\s+)?)?(?:(?:are|were|is)\s+)?(?:all|each)\b",
                             after)
                 or re.search(r"\b(?:values?|gaps?)\s+all\b|\ball\s+(?:lie|are|fall)\b", near[:max(0, end - lo) + 40]))
    sentence = text[s_start:s_end].lower()
    composition = bool(re.search(r"count\s+as\s+one\s+(?:compound|composition)|\bcompounds?\s+whose\b|"
                                 r"\bname\s+the\s+\w+\s+compounds\b", text.lower()))
    return ("every" if every else "any"), ("composition" if every and composition else "record")


_RELATIVE_OR_NOUN_RE = re.compile(
    r"\b(?:with|having|whose|that|which|where|of\s+the|devices?|cells?|materials?|compounds?|compositions?|records?|"
    r"samples?|emitters?|monolayers?|mofs?|perovskites?|thermoelectrics?|molecules?|entries)\b|물질|소자|재료|샘플|레코드|화합물",
    re.I)


def _every_before(pre: str) -> bool:
    """Whether a quantifier stands right before the quantity ("every band gap", "all of whose").

    "all materials with a band gap" and "every material whose band gap" quantify the population
    instead."""
    if re.search(r"\ball\s+of\s+(?:whose|its|their)\s+$", pre, re.I):
        return True
    m = None
    for m in re.finditer(r"\b(?:every|all|each)\b|모든", pre, re.I):
        pass
    if m is None:
        return False
    between = pre[m.end():]
    if re.search(r"[,:;]", between) or re.match(r"\s*of\b|\s*(?:sources?|collections?|databases?|"
                                                r"restricted|limited|filtered|bounded|constrained)\b", between, re.I):
        return False  # "for all of them, band gap ...", "from all sources": the population
    if re.search(r"\b(?:them|these|those|it)\s+$", pre[:m.start()], re.I):
        return False  # "list them all"
    if re.match(r"\s*(?:positive|non-?zero|(?:recorded|present|reported|available)\s*(?:,|and\b))", between, re.I):
        return False  # "PCE, Voc, Jsc and FF all positive ...": the J-V screen's
    if m.group(0).lower() == "all" and re.search(r"\b(?:list|show|give|find|return|get|print|name)\s+(?:me\s+)?(?:the\s+)?$",
                                                  pre[:m.start()], re.I):
        return False  # "list all band gaps above 2 eV": every value that is above 2 eV, one by one
    return len(between.split()) <= 2 and not _RELATIVE_OR_NOUN_RE.search(between)


_COMPLETE_RE = re.compile(
    r"full\s+j-?v(?:\s+set)?|complete\s+j-?v|(?:pce,?\s*voc,?\s*jsc,?\s*(?:and|&)\s*ff)\s*(?:all\s+)?"
    r"(?:present|recorded|reported|available)|all\s+four\s+(?:j-?v\s+)?metrics|완전한\s*j-?v", re.I)
#: "PCE, Voc, Jsc and FF all positive" in any order, "positive PCE, Voc, Jsc and FF", or the Korean
#: equivalent (the ledger's own wording, read back); "positive or zero" is not positive.
_FOUR_METRICS = r"(?:(?:pce|voc|jsc|ff)\s*(?:,|and|&|및|과|와)?\s*){4}"
_POSITIVE_RE = re.compile(
    rf"(?:{_FOUR_METRICS}|all\s+four\s+(?:j-?v\s+)?metrics\s*)(?:are\s+)?(?:all\s+)?(?:positive|non-?zero|above\s+zero|"
    rf"greater\s+than\s+zero|>\s*0(?![.\d])|모두\s*양수|모두\s*0\s*보다\s*큰)(?!\s*or\s+(?:zero|0|equal))"
    rf"|\bpositive\s+{_FOUR_METRICS}", re.I)
#: Where a comparison turns to its next population: each side's own light and completeness are read
#: apart.
_COMPARED_SIDE_RE = re.compile(r"\b(?:against|versus|vs\.?|compared\s+(?:with|to))\b|대비|(?:와|과)\s*비교", re.I)


def _device_conditions(text: str, segments) -> List[Condition]:
    from materials.services.device_policy import CellScreen
    out = []
    parts = []
    for start, end in segments:
        # "MP cells under 100 mW/cm2 against JARVIS cells under 100 mW/cm2": each side's own.
        cuts = [start] + [start + m.start() for m in _COMPARED_SIDE_RE.finditer(text[start:end])] + [end]
        parts += list(zip(cuts, cuts[1:]))
    for start, end in parts:
        chunk = text[start:end]
        screen = CellScreen.from_text(chunk)
        if (screen is not None and screen.require_full_jv) or _POSITIVE_RE.search(chunk):
            m = re.search(r"positive\s+PCE", chunk, re.I) or _POSITIVE_RE.search(chunk)
            out.append(Condition("completeness", completeness="positive",
                                 span=(start + m.start(), start + m.end()) if m else (start, end)))
        elif _COMPLETE_RE.search(chunk):
            m = _COMPLETE_RE.search(chunk)
            out.append(Condition("completeness", completeness="recorded",
                                 span=(start + m.start(), start + m.end())))
        if screen is not None and screen.irradiance_mw_cm2 is not None:
            m = re.search(r"\d+(?:\.\d+)?\s*mW|\bsuns?\b|\bAM\s*1\.5", chunk, re.I)
            out.append(Condition("illumination", irradiance_mw_cm2=screen.irradiance_mw_cm2,
                                 span=(start + m.start(), start + m.end()) if m else (start, end)))
    return out


#: Organic A-site cations written as letters in a formula (MAPbI3, FA0.85Cs0.15PbI3); set by the
#: caller with the corpus's own token list.
_CATION_TOKENS: Tuple[str, ...] = ()


#: Abbreviations people write for quantities and methods that also spell element symbols
#: ("FF" = F F, "WF" = W F, "SOC" = S O C): never a composition.
_NOT_FORMULAS = {"FF", "WF", "SOC", "CB", "VB", "IP", "CBM", "VBM", "PV", "IR", "UV", "NIR", "PF", "PL", "PCE", "EQE",
                 "IQE", "TE", "OLED", "LED", "CV", "IV", "JV", "NMR", "XRD", "SEM", "TEM", "DOS", "PBE", "HSE", "GW", "BSE",
                 # "ID" and "IDs" are identifier nouns, not I + Ds
                 "TCO", "HTL", "ETL", "HTM", "ETM", "ITO", "FTO", "ID", "IDs"}


def _is_formula(token: str, is_element: Callable[[str], bool]) -> bool:
    """Two or more constituents, each an element symbol or an organic-cation token."""
    if token in _NOT_FORMULAS:
        return False
    cations = "|".join(sorted(_CATION_TOKENS, key=len, reverse=True))
    pattern = (rf"(?:{cations})(?![a-z])|[A-Z][a-z]?" if cations else r"[A-Z][a-z]?")
    parts = re.findall(pattern, re.sub(r"[\d.]+", "", token))
    if "".join(parts) != re.sub(r"[\d.]+", "", token):
        return False
    return len(parts) >= 2 and all(p in _CATION_TOKENS or is_element(p) for p in parts)


def _exact_formulas(text: str, is_element) -> List[Condition]:
    out = []
    patterns = (
        rf"\b(?:plain|undoped|pristine|pure|unsubstituted|unalloyed)\s+(?P<f>{_FORMULA_RE.pattern})",
        rf"(?P<f>{_FORMULA_RE.pattern})[^.?!\n]{{0,40}}\(?\s*no\s+dopants?\b",
        rf"도핑\s*(?:안\s*된|되지\s*않은|없는)\s*(?P<f>{_FORMULA_RE.pattern})",
    )
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            formula = m.group("f")
            if _is_formula(formula, is_element):
                out.append(Condition("exact_formula", formulas=(formula,), span=(m.start(), m.end())))
    seen, unique = set(), []
    for c in out:
        if c.formulas not in seen:
            seen.add(c.formulas)
            unique.append(c)
    return unique


_EL = r"[A-Z][a-z]?"


def _substitution(text: str, is_element, parent_hint: Optional[str]) -> List[Condition]:
    m = (re.search(rf"\b(?P<el>{_EL})[-\s]site[-\s]+doped\b(?:\s+(?!with\b|by\b)\w+)?(?:\s+(?:with|by))?[\s,:(]*"
                   rf"(?P<dops>{_EL}(?:\s*(?:/|,|or|and)\s*{_EL})+)", text)
         or re.search(rf"doped\s+on\s+the\s+(?P<el>{_EL})\s+site\s+with\s+(?P<dops>{_EL}(?:\s*(?:/|,|,?\s*or|and)\s*{_EL})*)",
                      text)
         or re.search(rf"(?P<el>{_EL})\s*자리에\s*(?P<dops>{_EL}(?:\s*(?:/|,)\s*{_EL})*)\s*도핑", text))
    if not m or not is_element(m.group("el")):
        return []
    dopants = tuple(d for d in re.findall(_EL, m.group("dops")) if is_element(d))
    if not dopants:
        return []
    formulas = [f for f in _FORMULA_RE.findall(text) if _is_formula(f, is_element)]
    parent = formulas[0] if formulas else parent_hint
    allowed = bool(re.search(r"co-?doping\s+(?:\w+\s+)?(?:elsewhere\s+)?(?:is\s+)?(?:fine|ok|okay|allowed)|"
                             r"다른\s*자리[^.?!]{0,20}(?:괜찮|허용)", text, re.I))
    return [Condition("substitution", substitution={"parent": parent, "replaced_element": m.group("el"),
                                                    "dopants_any_of": dopants,
                                                    "co_substitution_allowed": allowed},
                      span=(m.start(), m.end()))]


_MEASURED_RE = re.compile(r"\b(?:measured|experimental(?:ly)?|observed)\b|실측|실험", re.I)
#: "measured at 100 mW/cm2", "measured under one sun": the condition of a measurement, not a
#: request for measured rather than computed values.
_MEASURED_CONDITION_RE = re.compile(r"^\s*(?:at|under|in|with|using|on|by|from|near|around|between|over|across|as|for|"
                                    r"to)\b", re.I)
#: "experimental methods", "experimentally synthesized": about how something was made or studied,
#: not a request for measured values.
_NOT_A_VALUES_FILTER_RE = re.compile(
    r"^\s*(?:methods?|techniques?|set-?ups?|procedures?|protocols?|studies|study|work|papers?|literature|"
    r"synthesi[sz](?:ed|is)|made|prepared|grown|realized|reported|fabricated|obtained|discovered|known)\b", re.I)
#: A measured population added to the answer: "experimental values too / if any / as well".
_ADDITIVE_AFTER_RE = re.compile(
    r"^\s*(?:값|치|데이터|values?|data|ones?|numbers?|results?|measurements?|gaps?)?\s*"
    r"(?:도|까지|이나|(?:이|가)\s*있으면|있으면|if\s+(?:any|there\s+are\s+any|available|recorded)|as\s+well|too\b)", re.I)
_ADDITIVE_BEFORE_RE = re.compile(
    r"\b(?:also|plus|including|as\s+well\s+as|along\s+with|together\s+with|and\s+(?:any|the|also))\s+"
    r"(?:the\s+|any\s+)?$", re.I)
_COMPUTED_WORD_RE = re.compile(r"\b(?:dft|calculated|computed|predicted|theoretical|theory|simulated)\b|계산", re.I)
#: A computed word that excludes ("not a computed one", "excluding DFT values"): it asks for
#: measured values rather than cancelling the request.
_EXCLUDING_BEFORE_RE = re.compile(r"(?:\bnot(?:\s+from)?|\bno|\bnon|\bexcluding|\bexclude|\bwithout|\bexcept|\brather\s+than|"
                                  r"\binstead\s+of|\bignore|\bignoring|\bother\s+than|\bbut\s+not)[-\s]+"
                                  r"(?:a\s+|an\s+|the\s+|any\s+)?$", re.I)
_EXCLUDING_AFTER_RE = re.compile(r"^[-\s]*(?:calculated\s+|predicted\s+)?(?:values?|ones?|data|gaps?|results?|numbers?|값)?\s*"
                                 r"(?:은|는|이|가)?\s*(?:말고|제외|빼고|아닌|없이|excluded|removed|dropped|left\s+out|"
                                 r"not\s+allowed|disallowed|aside)", re.I)


def _measured_only(text: str, bounds) -> List[Condition]:
    found: List[Condition] = []
    for m in _MEASURED_RE.finditer(text):
        s_start, s_end = _sentence_of(bounds, m.start())
        sentence = text[s_start:s_end]
        if re.search(r"\b(?:not|non|un)[-\s]?$", text[max(s_start, m.start() - 5):m.start()], re.I):
            continue
        counting = re.search(r"\bhow\s+many\b|\bwhich\b|\bwhat\s+(?:number|fraction|share)\b|몇", sentence, re.I)
        if re.search(r"\b(?:is|are|was|were)\s+(?:that|it|this|these|those|the\s+\w+)\s+$",
                     text[max(s_start, m.start() - 25):m.start()], re.I) \
                or re.match(r"\s*or\b", text[m.end():m.end() + 5], re.I) \
                or (not counting and re.match(r"\s*\?", text[m.end():m.end() + 5])):
            # "is that measured?", "measured or computed": asks which, restricts nothing -- but "how
            # many of them are measured?" counts the measured ones.
            continue
        if m.group(0).lower() in ("measured", "observed") and _MEASURED_CONDITION_RE.match(text[m.end():m.end() + 30]):
            continue
        if _NOT_A_VALUES_FILTER_RE.match(text[m.end():m.end() + 30]):
            continue
        if (_ADDITIVE_AFTER_RE.match(text[m.end():m.end() + 25])
                or _ADDITIVE_BEFORE_RE.search(text[max(s_start, m.start() - 30):m.start()])) \
                and not re.match(r"\s*(?:값|values?|data)?\s*(?:만|only)", text[m.end():m.end() + 12], re.I) \
                and not re.search(r"\bonly\s+$", text[max(s_start, m.start() - 12):m.start()], re.I):
            # "and experimental values if any", "as well as measured ones": a population added to
            # the answer, not the only one. "only measured" still restricts.
            continue
        found.append(Condition("measured_only", span=(m.start(), m.end())))
    for m in _COMPUTED_WORD_RE.finditer(text):
        # "excluding DFT values", "non-computed": measured values only.
        if _excluding(text, m.start(), m.end()) and not any(abs(c.span[0] - m.start()) < 40 for c in found):
            found.append(Condition("measured_only", span=(m.start(), m.end())))
    return found  # one per mention: each binds to its own population


_CONTRAST_RE = re.compile(r"\bcompare\b|\bvs\.?\b|versus|difference|differ|비교|차이", re.I)


def _excluding(text: str, start: int, end: int) -> bool:
    return bool(_EXCLUDING_BEFORE_RE.search(text[max(0, start - 30):start])
                or _EXCLUDING_AFTER_RE.match(text[end:end + 20]))


def _computed_contrasts(text: str, bounds, conditions: List[Condition], bind) -> List[Condition]:
    """Return the measured-only conditions that a 'computed' word cancels.

    A word cancels one in the same sentence, or anywhere in a question that compares the two kinds
    ("DFT gaps ... compare with the experimental gap" asks for both), but only when it binds to the
    same population: "measured MP materials ... against computed OQMD materials" keeps MP's
    measured-only. `bind` maps a span to its populations (None = every population)."""
    # Each word with the populations it qualifies, or None.
    words = [(m.start(), m.end(), bind(m.start(), m.end())) for m in _COMPUTED_WORD_RE.finditer(text)
             if not _excluding(text, m.start(), m.end())]
    if not words:
        return []
    contrast = bool(_CONTRAST_RE.search(text))
    cancelled = []
    for c in conditions:
        if c.kind != "measured_only":
            continue
        sentence = _sentence_of(bounds, c.span[0])
        for start, _, owner in words:
            same_population = owner is None or c.population is None or c.population in owner
            if same_population and (contrast or _sentence_of(bounds, start) == sentence):
                cancelled.append(c)
                break
    return cancelled


_UNNAMED_ENERGY_RE = re.compile(
    rf"(?:{_APPROX_WORD}\s*)?(?:between\s+(?P<a>{_NUM})\s*(?:m?eV\s*)?and\s*(?P<b>{_NUM})"
    rf"|(?P<a2>{_NUM})\s*(?:m?eV\s*)?(?:-|–|~|to|에서|부터)\s*(?P<b2>{_NUM}))\s*(?P<u>m?eV)(?![\w/])(?:\s*(?:사이|까지))?"
    rf"|(?P<w>{_LO_INCL}|{_HI_INCL}|{_LO_STRICT}|{_HI_STRICT})\s*(?P<c>{_NUM})\s*(?P<cu>m?eV)(?![\w/])"
    rf"|(?P<p>{_NUM})\s*(?P<pu>m?eV)\s*(?P<pw>이상|이하|미만|초과)"
    rf"|{_APPROX_WORD}\s*(?P<v>{_NUM})\s*(?P<vu>m?eV)(?![\w/])"
    rf"|(?P<k>{_NUM})\s*(?P<ku>m?eV)\s*(?:정도|가량|내외|쯤|안팎)", re.I)
#: What may stand between a quantity and its value: "electron affinity lies between", "HOMO of
#: about".
_LINKING_WORDS = (r"(?:\s|is|are|was|were|lies?|falls?|sits?|ranges?|of|between|from|in|around|about|approximately|"
                  r"roughly|near|close\s+to|below|under|above|over|less|more|greater|lower|higher|at\s+(?:least|most)|"
                  r"than|within|the\s+range|~|≈)*")
#: Words around a noun phrase that are not part of it.
_FUNCTION_WORDS = (r"with|whose|having|has|have|and|or|of|for|in|on|at|to|that|which|the|an?|its|their|is|are|was|were|"
                   r"lies?|falls?|sits?|ranges?|between|from|around|about|approximately|roughly|near|close|below|under|"
                   r"above|over|less|more|greater|lower|higher|least|most|than|within|range|show|list|give|find|me|"
                   r"any|all|some|only|those|ones?|중|중에서|에서|의|은|는|이|가|을|를|로|으로|인|인것|것")
#: The materials whose bare energy is their band gap: "absorbers around 1.3-1.5 eV".
_GAP_SUBJECT_RE = re.compile(r"absorb\w*|solar|photovoltaics?|pv|semiconduct\w*|perovskites?|흡수\w*|태양\w*|반도체|"
                             r"페로브스카이트", re.I)
#: Nouns that name a population, not a quantity: "materials between 1.3 and 1.5 eV" names no
#: quantity.
_GENERIC_NOUNS_RE = re.compile(r"materials?|compounds?|candidates?|samples?|records?|entries|systems?|structures?|"
                               r"물질|소재|재료|화합물|샘플", re.I)


def _qualified_noun(text: str, start: int, end: int) -> Tuple[str, str]:
    """Return the noun phrase an energy value qualifies and where it stands ("after" or "before").

    The phrase is written right after the value as an adjective ("10-20 meV Urbach tail",
    "1.3-1.5 eV absorbers") or right before it past linking words ("absorbers around 1.3-1.5 eV",
    "LO phonons above 10 meV"). ("", "") when there is none."""
    word = r"[A-Za-z가-힣][\w-]*"
    after = re.match(rf"\s*((?:{word}\s*){{1,3}})", text[end:end + 40])
    if after:
        words = []
        for w in after.group(1).split():
            if re.fullmatch(_FUNCTION_WORDS, w, re.I):
                break
            words.append(w)
        if words:
            return " ".join(words), "after"
    before = re.sub(rf"(?:\s|[,:~≈(]|(?<![\w-])(?:{_FUNCTION_WORDS})(?![\w-]))*$", "", text[max(0, start - 60):start],
                    flags=re.I)
    words = []
    for w in reversed(before.split()):
        if re.fullmatch(_FUNCTION_WORDS, w, re.I) or re.search(r"\d", w) or len(words) == 3:
            break
        words.insert(0, w.strip(",;:()"))
    return " ".join(words), "before" if words else ""


def _unnamed_energy_windows(text: str, consumed, mentions, chain_windows=(), context: str = "") -> List[Condition]:
    """Read energy windows that no named quantity claims, by the noun each one qualifies.

    An absorber's or a semiconductor's window ("lead-free absorbers around 1.3-1.5 eV") is its band
    gap, and is labelled so. Another noun's ("LO phonons above 10 meV", "Urbach tail") belongs to a
    quantity LAMP does not store and is refused by name. With no noun, or a population noun
    ("materials between 1.3 and 1.5 eV"), the quantity is unnamed and refused: never dropped, which
    would list materials as if they met it, and never assumed to be the band gap. After a turn with
    a window on an eV quantity, a bound naming no quantity is that quantity's (`_bare_bounds`), so
    only a named one is read here."""
    from .rag import STORED_UNITS
    eV_chain = any(STORED_UNITS.get(t[0]) == "eV" for t in chain_windows)
    taken = list(consumed) + [(m.start, m.end) for m in mentions]
    found: List[Condition] = []
    for m in _UNNAMED_ENERGY_RE.finditer(text):
        if any(a < m.end() and m.start() < b for a, b in taken):
            continue
        if any(0 <= m.start() - q.end <= 40 and re.fullmatch(_LINKING_WORDS, text[q.end:m.start()], re.I) for q in mentions):
            continue  # "electron affinity lies between 3.65 and 4.05 eV": that quantity's value
        if any(0 <= q.start - m.end() <= 3 for q in mentions):
            continue  # "1.5 eV band gap": that quantity's, read with it
        noun, _ = _qualified_noun(text, m.start(), m.end())
        words = noun.split()
        cut = max((k + 1 for k, w in enumerate(words[:-1]) if _GAP_SUBJECT_RE.fullmatch(w) or _GENERIC_NOUNS_RE.fullmatch(w)),
                  default=0)
        noun = " ".join(words[cut:])  # "perovskite Urbach energy": the quantity is "Urbach energy"
        head = noun.split()[-1] if noun else ""
        values = [_num(m[g]) for g in ("a", "b", "a2", "b2", "c", "p", "v", "k") if m[g] is not None]
        if head and _GAP_SUBJECT_RE.fullmatch(head) and min(values) >= 0:
            target, note = ("bandgap",), "read as the band gap"
        elif head and not _GENERIC_NOUNS_RE.fullmatch(head):
            target, note = (f"unstored:{noun}",), ""
        elif not head and not chain_windows and _GAP_SUBJECT_RE.search(f"{context} {text}") and min(values) >= 0:
            target, note = ("bandgap",), "read as the band gap"
        else:
            target, note = ("unstored:unnamed eV quantity",), ""
        if chain_windows and (eV_chain or not target[0].startswith("unstored:") or "unnamed" in target[0]):
            continue  # a follow-up's bound: _bare_bounds reads it as the conversation's quantity
        mention = _Mention(m.start(), m.end(), target)
        if m["a"] is not None or m["a2"] is not None:
            lo, hi = (m["a"], m["b"]) if m["a"] is not None else (m["a2"], m["b2"])
            window = replace(_window(mention, Bound(_num(lo), True), None, m["u"], 0), hi=Bound(_num(hi), True))
        elif m["w"] is not None or m["p"] is not None:
            if m["w"] is not None:
                side, inclusive, _ = _comparator(m["w"], text[:m.start("w")])
                value, unit = m["c"], m["cu"]
            else:
                side, inclusive = ("lo" if m["pw"] in ("이상", "초과") else "hi"), m["pw"] in ("이상", "이하")
                value, unit = m["p"], m["pu"]
            bound = Bound(_num(value), inclusive)
            window = _window(mention, bound if side == "lo" else None, bound if side == "hi" else None, unit, 0)
        else:
            window = _approximate(mention, _num(m["v"] or m["k"]), m["vu"] or m["ku"])
        written = m.group(0).strip()
        found.append(replace(window, span=(m.start(), m.end()), qualitative=(
            f"the question's '{written}' names no quantity; {note}" if note else
            "" if target[0] != "unstored:unnamed eV quantity" else f"the question's '{written}' names no quantity")))
    return found


_QUALITATIVE = (
    (re.compile(r"\b(?:zero[-\s]?gap|gapless|metallic)\b", re.I), "zero-gap", None, Bound(0.0, True)),
    (re.compile(r"\bsemiconducting\b", re.I), "semiconducting", Bound(0.0, False), None),
)

#: Families that, named anew, mean a new population ("now list perovskites" after "C2DB
#: monolayers").
_DISTINCT_FAMILIES = {"mof", "emitter", "monolayer", "perovskite", "thermoelectric"}

#: Nouns that name a population on their own ("how many MOFs in this DB", "perovskite device
#: records").
_FAMILY_NOUNS = {"mof", "emitter", "monolayer", "perovskite", "thermoelectric", "device", "record", "sample",
                 "material", "compound", "composition"}

#: A temperature qualifying the cells: "solar cells measured at 700 K", "devices tested at", "LEDs
#: at" (the cells right before it) and, in Korean word order, the cells right after it.
_DEVICE_AT_RE = re.compile(r"(?:\bdevices?|(?<!unit\s)\bcells?|\bo?leds?|소자|태양\s*전지)\s+"
                           r"(?:(?:that|which)\s+)?(?:(?:were|are|was|is)\s+)?"
                           r"(?:(?:measured|tested|operated|operating|characteri[sz]ed|recorded|aged|held|run)\s+)?$", re.I)
_AT_DEVICE_RE = re.compile(r"\s*(?:에서)?\s*(?:측정|시험|동작|구동|평가)(?:한|된|하는|되는)\s*(?:[가-힣A-Za-z]+\s+){0,2}?"
                           r"(?:[가-힣]*태양\s*전지|[가-힣]*소자|(?<![가-힣])셀(?![가-힣]{0,1}레)|O?LED|[가-힣]*전지)(?![가-힣]*\s*소재)", re.I)

_NOUNS = (
    ("device", r"\bdevices?\b|소자|태양\s*전지|(?<![가-힣])셀(?![가-힣]{0,1}레)|\bcells?\b(?!\s+(?:materials?|compositions?))"),
    ("material", r"\bmaterials?\b|물질|재료"),
    ("compound", r"\bcompounds?\b"),
    ("composition", r"\bcompositions?\b|조성"),
    ("record", r"\brecords?\b|레코드"),
    ("sample", r"\bsamples?\b|샘플"),
    ("emitter", r"\bemitters?\b"),
    ("monolayer", r"\bmonolayers?\b"),
    ("mof", r"\bmofs?\b|metal[-\s]organic\s+frameworks?"),
    ("perovskite", r"\bperovskites?\b|페로브스카이트"),
    ("thermoelectric", r"\bthermoelectrics?\b|열전"),
)


def _noun(text: str) -> str:
    best = None
    for name, pattern in _NOUNS:
        m = _ascii(pattern).search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), name, m.end())
    if best is None:
        return ""
    if best[1] in _DISTINCT_FAMILIES and re.match(r"\s*(?:solar\s+)?(?:cells?|devices?|o?leds?)\b|\s*(?:태양\s*전지|소자|셀)",
                                                  text[best[2]:], re.I):
        return "device"  # "perovskite solar cells": the cells are the head noun
    return best[1]


def _populations(text: str, is_element, sources_in) -> Tuple[List[Population], List[Tuple[int, int]]]:
    """Return (populations, their subject spans; empty for a single population).

    One population, unless the question contrasts distinct subjects ("X with that for Y", "X vs
    Y", "compare X and Y", "SRC and SRC") or counts each subject separately ("how many MP ... and
    how many OQMD ...")."""
    compare = re.search(r"\bcompare[sd]?\b|\bvs\.?\b|versus|\bagainst\b|비교", text, re.I)
    named = _named_subjects(text, is_element)
    sources = list(dict.fromkeys(sources_in(text)))
    if compare and (len(named) >= 3 or len(sources) >= 3):
        # "Compare MP, OQMD and JARVIS ...", "Compare ZnO, GaN and CdS": every one.
        noun = _noun(text)
        pops = ([Population(formulas=(f,), noun=noun, span=(a, b)) for f, a, b in named] if len(named) >= 3 else
                [Population(sources=(src,), noun=noun, span=(0, len(text))) for src in sources])
        return pops, [p.span for p in pops]
    segments = _count_segments(text)
    if len(segments) > 1:
        # "How many HOPV cells ..., and how many PerovskiteDB cells ...": one population per
        # counted source.
        per = [sources_in(text[a:b]) for a, b in segments]
        if all(len(x) == 1 for x in per) and len({x[0] for x in per}) == len(per):
            noun = _noun(text)
            pops = [Population(sources=(x[0],), noun=noun, span=(0, len(text))) for x in per]
            return pops, [p.span for p in pops]
    contrast = re.compile(
        rf"(?P<a>{_FORMULA_RE.pattern}|[A-Z][A-Za-z0-9_-]+)\s+(?:with\s+that\s+(?:for|of)|vs\.?|versus|compared\s+"
        rf"(?:with|to)|and|or)\s+(?P<b>{_FORMULA_RE.pattern}|[A-Z][A-Za-z0-9_-]+)")
    for m in contrast.finditer(text):
        a, b = m.group("a"), m.group("b")
        if a == b:
            continue
        a_src, b_src = sources_in(a), sources_in(b)
        a_f, b_f = _is_formula(a, is_element), _is_formula(b, is_element)
        if not ((a_src and b_src) or (a_f and b_f)):
            continue
        if a_f and not re.search(r"\bcompare\b|\bvs\.?\b|versus|with\s+that\s+(?:for|of)|비교", text, re.I):
            continue
        noun = _noun(text)
        first = Population(sources=tuple(a_src), formulas=(a,) if a_f else (), noun=noun,
                           span=(m.start("a"), m.end("a")))
        second = Population(sources=tuple(b_src), formulas=(b,) if b_f else (), noun=noun,
                            span=(m.start("b"), m.end("b")))
        return [first, second], [first.span, second.span]
    # "Compare MAPbI3 cells with FF above 70 % against FAPbI3 cells with FF above 80 %": a
    # comparison naming two or more distinct compositions, not only side by side.
    if compare:
        if len(named) >= 2:
            noun = _noun(text)
            pops = [Population(formulas=(f,), noun=noun, span=(a, b)) for f, a, b in named]
            return pops, [p.span for p in pops]
        # ... or two or more sources, wherever they sit ("MP materials with ... against OQMD
        # materials with ...").
        if len(sources) >= 2:
            noun = _noun(text)
            pops = [Population(sources=(src,), noun=noun, span=(0, len(text))) for src in sources]
            return pops, [p.span for p in pops]
    return [Population(sources=tuple(sources_in(text)), noun=_noun(text), span=(0, len(text)))], []


def _named_subjects(text: str, is_element) -> List[Tuple[str, int, int]]:
    named: List[Tuple[str, int, int]] = []
    for m in _FORMULA_RE.finditer(text):
        f = m.group(1)
        if _is_formula(f, is_element) and f not in [n for n, _, _ in named]:
            named.append((f, m.start(), m.end()))
    return named


_COORDINATION_RE = re.compile(
    r"\s*(?:(?:materials?|compounds?|records?|cells?|devices?|samples?|monolayers?|emitters?|물질|소자)\s*)?"
    r"(?:,|and|or|&|vs\.?|versus|및|과|와)\s*(?:how\s+many\s+|the\s+)?$", re.I)


#: "all" of the populations as subjects: "..., all with", "all of them having", "all three" -- not
#: "all band gap values" (the values' quantifier) nor "both"/"each" (the coordinated pair's).
_ALL_OF_THEM_RE = re.compile(
    r"\b(?:all|every\s+one)(?:\s+of\s+(?:them|these|those|which|whom))?(?:\s+(?:three|four|five|\d+|materials?|compounds?|"
    r"records?|cells?|samples?|sets?|sources?|collections?))?\s*(?:,|:|with\b|having\b|whose\b|that\b|which\b|must\b|"
    r"should\b|have\b|has\b|are\b|restricted\b|limited\b|filtered\b|bounded\b|constrained\b)|\ball\s+(?:three|four|five|"
    r"\d+)\b|셋\s*다|세\s*(?:곳|개)\s*모두|모두|전부", re.I)


#: "both" / "each" quantifying the subjects: after a comma or "where/and/with", or "both of them",
#: "each one", "in both cases" -- never "each material", "whose records each have" (per item), nor
#: "with both a density ... and ..." (one subject's two quantities).
_BOTH_RE = re.compile(
    r"(?:(?:^|[,;:—])\s*|\b(?:where|and|with)\s+)(?:both|each)\b(?!\s+(?:a|an|the|materials?|records?|compounds?|cells?|"
    r"devices?|samples?|entries|entry|items?|values?|structures?|compositions?)\b)|\b(?:both|each)\s+(?:of\s+(?:them|"
    r"which|whom)|one)\b|\bin\s+both\s+cases\b|둘\s*다|각각", re.I)
_JOINED_RE = re.compile(
    r"\s*(?:(?:materials?|compounds?|records?|cells?|devices?|samples?|monolayers?|emitters?|물질|소자)\s*)?"
    r"(?:,|and|or|&|vs\.?|versus|with|to|against|및|과|와|를|을)\s*(?:the\s+)?$", re.I)


def _bound_population(text, bounds, occurrences, c, n_populations) -> Optional[int]:
    """Return the one population a condition belongs to, or None for several or all."""
    owners = _population_owners(text, bounds, occurrences, c)
    return next(iter(owners)) if owners is not None and len(owners) == 1 else None


def _population_owners(text, bounds, occurrences, c) -> Optional[set]:
    """Return the populations a condition belongs to.

    These are the subject named nearest before it in its sentence and every subject coordinated
    with it ("MP and OQMD materials with ..." -> {MP, OQMD}, not every population); None for a
    subject-less clause (all of them)."""
    s_start, s_end = _sentence_of(bounds, c.span[0])
    inside = [(a, b, i) for a, b, i in occurrences if s_start <= a < s_end]
    if c.kind in ("measured_only", "computed"):
        owners = _adjective_owners(text, inside, c.span)
        if owners is not None:
            return owners
    prior = [(a, b, i) for a, b, i in inside if a <= c.span[0]]
    if not prior:
        # Stated before every subject ("Among materials with band gap above 2 eV, compare MP and
        # OQMD against JARVIS", "above 2 eV: count MP and OQMD separately"): every population's.
        return None
    between = text[prior[-1][1]:c.span[0]]
    named = {i for _, _, i in prior}
    group = [prior[-1]]
    for a, b, i in reversed(prior[:-1]):
        if i != group[-1][2] and _COORDINATION_RE.fullmatch(text[b:group[-1][0]] or " "):
            group.append((a, b, i))
        else:
            break
    joined = [prior[-1]]
    for a, b, i in reversed(prior[:-1]):
        # "Compare MP with OQMD": a compared pair is joined too.
        if i != joined[-1][2] and _JOINED_RE.fullmatch(text[b:joined[-1][0]] or " "):
            joined.append((a, b, i))
        else:
            break
    every = _ALL_OF_THEM_RE.search(between)
    if every and len({i for _, _, i in joined}) > 1 and not [o for o in inside if o[0] > c.span[0]] \
            and re.search(r"[,:;]", between[:every.start() + 1]):
        # "Compare JARVIS with MP and OQMD, all with band gap above 2 eV": every subject's, closing
        # a list; not "for MP all materials with ..., for OQMD ...", one subject's own clause.
        # "Both"/"each" are the coordinated pair's, and "all band gap values" is the quantity's own
        # quantifier.
        return None
    both = _BOTH_RE.search(between)
    after_both = text[prior[-1][1] + both.end():c.span[0]] if both else ""
    # "with both a density above 5 g/cm3 and a formation energy ...": one subject's quantities,
    # "with both" joining them by and/plus/as well as. After a comma or "and/where" ("..., both
    # exhibiting ZT above 1"), "both" quantifies the subjects.
    quantities = bool(both) and re.match(r"(?i)with\s", both.group(0)) is not None and re.search(
        r"\b(?:and|plus|as\s+well\s+as)\b|&", text[prior[-1][1] + both.end():], re.I) is not None
    if len({i for _, _, i in group}) == 1 and len(named) > 1 and both and not quantities and ";" not in after_both:
        # "MP materials with ... and OQMD materials with ..., both having ZT above 1": the subjects
        # listed with "and" up to the last one, each with its own qualifier; never "against JARVIS"
        # before them, nor "with both band gap ... and density" (one subject's two quantities).
        listed = [prior[-1]]
        for a, b, i in reversed(prior[:-1]):
            gap = text[b:listed[-1][0]]
            if i != listed[-1][2] and re.search(r"(?:,|\band|및|과|와|이고|이며|하고|그리고|인데)\s*(?:the\s+)?$", gap, re.I) and not re.search(
                    r"\b(?:against|versus|vs\.?|compared|than)\b|대비|비교", gap, re.I):
                listed.append((a, b, i))
            else:
                break
        if len({i for _, _, i in listed}) > 1:
            return {i for _, _, i in listed}
        # No "and" list: "both" is the last two subjects named ("MP materials with ... versus OQMD
        # materials, both with ..."), "each" every subject named.
        last_two = list(dict.fromkeys(i for _, _, i in reversed(prior)))[:2]
        return set(last_two) if re.match(r"both|둘", both.group(0), re.I) else named
    return {i for _, _, i in group}


def _adjective_owners(text, inside, span) -> Optional[set]:
    """Return the subjects an evidence adjective ("measured", "computed") qualifies.

    That is the subject right before it ("ZnO (measured)"), else the subject right after it and
    every subject coordinated with that one ("the measured band gaps of ZnO and GaN"); None when
    neither."""
    # "ZnO (measured)", "MP materials (measured)": the subject right before it, its noun between.
    own = [(a, b, i) for a, b, i in inside if b <= span[0] and re.fullmatch(
        r"\s*(?:(?:materials?|compounds?|records?|cells?|devices?|samples?|monolayers?|values?|물질|소자)\s*)?[\s(]*",
        text[b:span[0]], re.I)]
    if own:
        return {own[-1][2]}
    nxt = [(a, b, i) for a, b, i in inside if span[1] <= a <= span[1] + 20]
    if not nxt:
        return None
    group, last = {nxt[0][2]}, nxt[0]
    for a, b, i in [o for o in inside if o[0] > nxt[0][0]]:
        if _COORDINATION_RE.fullmatch(text[last[1]:a]):  # "of MP materials and OQMD materials"
            group.add(i)
            last = (a, b, i)
        else:
            break
    return group


def _subject_occurrences(text: str, populations: List[Population], sources_in) -> List[Tuple[int, int, int]]:
    """Return every (start, end, population index) where a population's subject is named.

    Not only the contrast that introduced it: "Compare MP and OQMD. List materials in MP with ...
    List materials in OQMD with ..." binds each window to the source its own sentence names."""
    found = []
    for index, population in enumerate(populations):
        for formula in population.formulas:
            found += [(m.start(), m.end(), index) for m in re.finditer(re.escape(formula), text)]
        # ASCII tokens, so "MP" followed by a Korean particle still reads as the token MP.
        tokens = list(re.finditer(r"[A-Za-z0-9][A-Za-z0-9_-]*", text))
        for source in population.sources:
            for k, m in enumerate(tokens):
                if source in sources_in(m.group(0)):
                    found.append((m.start(), m.end(), index))  # the name itself ("MP")
                elif k + 1 < len(tokens) and source in sources_in(f"{m.group(0)} {tokens[k + 1].group(0)}") \
                        and source not in sources_in(tokens[k + 1].group(0)):
                    found.append((m.start(), tokens[k + 1].end(), index))  # a two-word name
    return sorted(set(found))


#: Where a COUNT operation starts; one definition for operations, device segments and intersection.
_COUNT_RE = re.compile(r"\bhow\s+many\b|\bnumber\s+of\b|\bcount\b(?!\s+(?:only|as)\b)|몇\s*(?:개|건|가지)|"
                       # Korean count nouns followed by a request verb or a question end
                       r"(?:개수|건수|총\s*수)(?=\s*(?:를|는|가|만)?\s*(?:알려|세|보여|말해|구해|확인|[?.!]|$))", re.I)


#: What may follow a Korean clause-closing count word: a copula or question ending, then an optional
#: request ("tell me", "I wonder").
_KO_COUNT_ENDING = (r"[\s?.!]*(?:(?:인지|있는지|되는지|일까|인가요?|입니까|이야|이에요|예요|야|요|고|나요?|죠)\s*)*"
                    r"(?:알려\s*(?:줘|주세요|줄래|줄\s*수\s*있어\?*)|말해\s*(?:줘|주세요)|궁금해요?|확인해\s*줘)?[\s?.!요]*")


def _elliptical(clause: str, sources_in) -> bool:
    """Whether a count clause names only its source ("and how many in OQMD?").

    "how many materials does OQMD hold in total?" asks something of its own."""
    rest = _COUNT_RE.sub(" ", clause)
    for source in sources_in(clause):
        rest = re.sub(re.escape(source), " ", rest, flags=re.I)
    rest = re.sub(r"\b(?:and|in|for|from|of|the|about|what|how|materials?|records?|cells?|compounds?|there|are|is)\b|"
                  r"에서|에는|에|은|는|이|가|도|야|요|개|몇|그리고|[,?.!:]", " ", rest, flags=re.I)
    return not re.search(r"[A-Za-z가-힣]", rest)


def _count_segments(text: str) -> List[Tuple[int, int]]:
    """Return one span per COUNT operation when there are several, else the whole text.

    An English count word opens its clause ("how many ..., and how many ..."); a Korean one closes
    it, so there the clause runs from the previous count word to this one."""
    counts = list(_COUNT_RE.finditer(text))
    if len(counts) < 2:
        return [(0, len(text))]
    if re.fullmatch(_KO_COUNT_ENDING, text[counts[-1].end():]) and text[:counts[0].start()].strip():
        ends = [m.end() for m in counts]
        return [(ends[i - 1] if i else 0, end if i + 1 < len(ends) else len(text)) for i, end in enumerate(ends)]
    starts = [m.start() for m in counts]
    return [(start, starts[i + 1] if i + 1 < len(starts) else len(text)) for i, start in enumerate(starts)]


def _operations(text: str, conditions: List[Condition]) -> List[Operation]:
    ops: List[Operation] = []
    if _COUNT_RE.search(text):
        for k, span in enumerate(_count_segments(text)):
            # "..., and how many of those also have ...": nested in the previous count.
            clause = re.split(r"[.?!]", text[span[0]:span[1]], maxsplit=1)[0]  # own sentence only
            nested = k > 0 and bool(_ANAPHORA_RE.search(clause) or re.search(
                r"\balso\s+(?:have|has|are|is|with|carry|show|meet)\b|\b(?:do|did|are|were)\s+they\b|\bthey\s+(?:also|have|are|"
                r"belong|come)\b|\bis\s+that\b|그\s*중|이\s*중", clause, re.I))
            ops.append(Operation("COUNT", span=span, nested_in=k - 1 if nested else None))
    if any(c.kind == "rank" for c in conditions):
        r = next(c for c in conditions if c.kind == "rank")
        ops.append(Operation("RANK", span=r.span))
    # The request verb of a Korean count ("tell me how many"), not a request for a list.
    asked = text
    for m in reversed(list(_COUNT_RE.finditer(text))):
        ending = re.match(r"\s*(?:를|은|는|가|만)?" + _KO_COUNT_ENDING, text[m.end():])
        asked = asked[:m.start()] + " " + asked[m.end() + (ending.end() if ending else 0):]
    if asked != text:
        # "How many binary oxides does OQMD list ...": the count's own verb.
        asked = re.sub(r"\b(?:does|do|did)\s+(?:\S+\s+){0,3}?(?:list|show|name)\b", " ", asked, flags=re.I)
    if re.search(r"\b(?:list|show|which|shortlist|recommend|name|find|give\s+me)\b|뽑아|추천|알려", asked, re.I):
        ops.append(Operation("SELECT", span=(0, len(text))))
    if re.search(r"\bcompare\b|\bvs\.?\b|versus|비교", text, re.I):
        ops.append(Operation("COMPARE", span=(0, len(text))))
    if not ops:
        ops.append(Operation("LOOKUP", span=(0, len(text))))
    count_ops = [o for o in ops if o.kind == "COUNT"]
    for index, condition in enumerate(conditions):
        for op in ops:
            if op.span[0] <= condition.span[0] < op.span[1] and (op.kind != "COUNT" or condition.kind != "rank"):
                op.conditions.append(index)
    if len(count_ops) > 1:
        for k, op in enumerate(count_ops):
            for index in op.conditions:
                conditions[index].local = True
                conditions[index].op = k
                conditions[index].op_parent = op.nested_in
    return ops


#: Quantities whose stored values carry a measurement temperature: a temperature next to one of them
#: is its own, even when cells are named right before it ("thermoelectric devices at 600 K with ZT
#: above 1").
TEMPERATURE_BEARING = frozenset({"ZT", "Seebeck_coefficient", "seebeck_p", "seebeck_n", "thermal_conductivity",
                                 "electrical_conductivity", "electrical_resistivity", "power_factor",
                                 "figure_of_merit_Z"})


#: Device words whose records are solar cells or LEDs -- rows that carry no temperature-bearing
#: quantity, so a temperature on them is the cells'; a thermoelectric device, a fuel cell or a
#: module is measured by its material's temperature-bearing quantities.
_CELL_DEVICE_RE = re.compile(r"solar|photovoltaic|\bpv\b|perovskite\s+(?:cells?|devices?)|"
                             r"(?<!fuel\s)(?<!thermoelectric\s)(?<!electrochemical\s)(?<!unit\s)\bcells?\b|"
                             r"\bo?leds?\b|light[-\s]emitting|태양|(?<![가-힣])셀(?![가-힣]{0,1}레)|발광", re.I)


def _temperature_assignments(text, temperatures, conditions, ranks, mentions, bounds) -> List[List[Tuple[str, ...]]]:
    """Return, per temperature, the owners whose measurement temperature it is.

    An owner is a quantity's names, ("device",) for the cells, or () when the temperature bounds
    nothing named.

    Two passes. A temperature attached to something belongs to it: inside or closing a quantity's
    phrase ("ZT at 600 K above 1", "band gap above 1 eV at 300 K and ..."), right after a
    quantity's name ("ranked by ZT at 600 K"), or on a device ("solar cells tested at 700 K");
    closing the last of a list, it belongs to each listed quantity ("ZT ... and Seebeck ... at
    600 K"). Every other temperature opens a scope: it bounds each quantity, windows and rankings
    alike, from it to the next opening temperature that no attached temperature has taken ("At
    600 K, top 5 by ZT with ..."), else each quantity before it ("..., both at 600 K")."""
    def owner(target):
        return ("device",) if target[0] in _DEVICE_METRIC_NAMES else target

    def each(targets):
        # Several quantities: each temperature-bearing one, else each one (reported as unbounded).
        targets = list(dict.fromkeys(owner(x) for x in targets))
        bearing = [x for x in targets if set(x) & TEMPERATURE_BEARING]
        return bearing or targets or [()]

    windows = [c for c in conditions if c.kind == "window" and c.target]
    quantities = sorted([(c.span[0], c.target) for c in windows] + [(c.span[0], c.target) for c in ranks if c.target])
    out: List[Optional[List[Tuple[str, ...]]]] = []
    # The quantity occurrences an attached temperature takes, by position, so the same quantity
    # elsewhere (another population's, another sentence's) is still open to another temperature.
    held: set = set()

    def take(targets, s_start, s_end, upto=None):
        held.update(pos for pos, x in quantities if s_start <= pos < s_end and (upto is None or pos <= upto)
                    and owner(x) in {owner(y) for y in targets})

    for t in temperatures:
        s_start, s_end = _sentence_of(bounds, t.span[0])
        clause = re.search(r",\s*(?:and\s+)?how\s+many\b|;|\band\s+how\s+many\b", text[t.span[1]:s_end], re.I)
        c_end = t.span[1] + clause.start() if clause else s_end
        mine = [c for c in windows if s_start <= c.span[0] < s_end]
        before = [c for c in mine if c.span[0] <= t.span[0]]
        inside = [c for c in before if t.span[0] < c.span[1]]
        found: Optional[List[Tuple[str, ...]]] = None
        if inside:
            found = [owner(inside[-1].target)]
            held.add(inside[-1].span[0])
        else:
            opening = text[t.span[0]:t.span[1]].rstrip().endswith("에서")
            closes = [c for c in before if 0 <= t.span[0] - c.span[1] <= 3
                      # A Korean temperature after a comma, or right before a quantity, opens.
                      and not (opening and (re.search(r"[,;]", text[c.span[1]:t.span[0]])
                                            or (any(t.span[1] <= q.span[0] <= t.span[1] + 3 for q in mine)
                                                and not re.match(r"\s*[,;]", text[t.span[1]:]))))]
            # Its list: the quantities since the previous temperature in its sentence ("band gap
            # below 1 eV at 300 K, ZT ... and Seebeck ... at 600 K": ZT and Seebeck) and since the
            # last comparison word ("MP at 300 K with ZT ... against OQMD with ZT ... and Seebeck
            # ... at 600 K" lists OQMD's two). An opening temperature before them owns them.
            previous = [o for o in temperatures[:len(out)] if s_start <= o.span[1] <= t.span[0]]
            turn = max([s_start] + [m.end() for m in _COMPARED_SIDE_RE.finditer(text, s_start, t.span[0])])
            floor = max(turn, previous[-1].span[1] if previous else s_start)
            listed = [c for c in before if c.target[0] not in _DEVICE_METRIC_NAMES and c.span[0] >= floor]
            # An opening temperature right before the list owns it ("at 300 K with Seebeck ... and
            # ZT ... at 600 K").
            opened = bool(previous) and out[temperatures.index(previous[-1])] is None and previous[-1].span[0] >= turn
            alone = bool(listed) and not opened and not any(out[j] is None and floor <= o.span[0] < t.span[0]
                                                            for j, o in enumerate(temperatures[:len(out)]))
            later_temps = [o for o in temperatures if t.span[1] <= o.span[0] < c_end]
            named = [m for m in mentions if s_start <= m.start and m.end <= t.span[0]
                     and re.fullmatch(r"\s*(?:(?:of|for|in)\s+[^\s,;]+\s*)?", text[m.end:t.span[0]], re.I)]
            bearing = [m.target for m in mentions if s_start <= m.start < s_end and set(m.target) & TEMPERATURE_BEARING]
            context = text[max(s_start, t.span[0] - 80):t.span[0]]
            base = max(s_start, t.span[0] - 60)
            before_device = None if re.search(r"\bunit\s+cells?\s*$", context, re.I) else _DEVICE_AT_RE.search(
                text[base:t.span[0]])
            after_device = _AT_DEVICE_RE.match(text, t.span[1])
            device = before_device or after_device
            if closes and closes[-1].target[0] in _DEVICE_METRIC_NAMES:
                found = [("device",)]  # "... and PCE above 20 % at 700 K": the cells'
                held.add(closes[-1].span[0])
            elif closes and alone and len({c.target for c in listed}) > 1 and closes[-1] in listed:
                # "ZT above 1 and Seebeck above 200 µV/K at 600 K[, ranked by power factor]": each
                # of them, with the rankings stated among or after them when no later temperature
                # takes those.
                rank_scope = [r for r in ranks if r.target and listed[0].span[0] <= r.span[0] < (
                    c_end if not later_temps else t.span[0])]
                found = each([c.target for c in listed] + [r.target for r in rank_scope])
                held.update(c.span[0] for c in listed)
                held.update(r.span[0] for r in rank_scope)
            elif closes:
                found = [owner(closes[-1].target)]
                held.add(closes[-1].span[0])
            elif named:
                found = [owner(named[-1].target)]
                # Only that occurrence: "OQMD materials ranked by ZT at 600 K" never takes MP's ZT
                # window.
                mine_q = [pos for pos, x in quantities if s_start <= pos <= t.span[0] and owner(x) == found[0]]
                held.update(mine_q[-1:])
            elif device:
                # The device word read in its own context: "solid oxide fuel cells operating at"
                # is no solar cell, and "solar cells and thermoelectric devices tested at" is the
                # thermoelectric devices'.
                lead = re.search(r"(?:[\w-]+\s+){0,2}$", text[s_start:base + before_device.start()]) if before_device else None
                near = (lead.group(0) + before_device.group(0)) if before_device else after_device.group(0)
                cell = _CELL_DEVICE_RE.search(near)
                found = [("device",)] if cell or not bearing else each(bearing)
                if found != [("device",)]:
                    take(found, s_start, s_end)
        out.append(found)
    for k, t in enumerate(temperatures):
        if out[k] is not None:
            continue
        s_start, s_end = _sentence_of(bounds, t.span[0])
        later = [o for j, o in enumerate(temperatures) if out[j] is None and t.span[1] <= o.span[0] < s_end]
        # Up to the next opening temperature, never past a comparison into another population.
        turn = next((m.start() for m in _COMPARED_SIDE_RE.finditer(text, t.span[1], s_end)), s_end)
        stop = min(later[0].span[0] if later else s_end, turn)
        free = [(pos, x) for pos, x in quantities if pos not in held and s_start <= pos < s_end]
        scope = [x for pos, x in free if t.span[1] <= pos < stop] or [x for pos, x in free if pos < t.span[0]]
        if scope:
            out[k] = each(scope)
            continue
        device_at = max([m.end() for m in _ascii(_NOUNS[0][1]).finditer(text, s_start, t.span[0])
                         if not re.search(r"\bunit\s+$", text[max(s_start, m.start() - 6):m.start()], re.I)]
                        + [x.end for x in mentions if x.device and s_start <= x.end <= t.span[0]] + [-1])
        out[k] = [("device",)] if device_at >= 0 and _CELL_DEVICE_RE.search(
            text[max(s_start, device_at - 40):device_at]) else [()]
    return out


def _read(text: str, *, property_patterns, is_element, sources_in, parent_hint=None, chain_windows=(),
          context: str = "") -> QuestionPlan:
    own, pasted = cut_pasted(text)
    bounds = _sentence_bounds(own)
    mentions = _mentions(own, property_patterns)
    conditions: List[Condition] = []
    consumed: List[Tuple[int, int]] = []
    phrase_end = 0  # where the previous quantity's window phrase ends
    for i, mention in enumerate(mentions):
        if mention.computed:
            continue
        limit = mentions[i + 1].start if i + 1 < len(mentions) else len(own)
        limit = min(limit, _sentence_of(bounds, mention.start)[1], mention.end + 60)
        hit = _window_after(own, mention, limit)
        number = re.search(r"[-+−]?\d", own[mention.end:])
        if hit and number and re.search(r"\band\b|\bor\b|,|&|(?<![가-힣])및(?![가-힣])",
                                         own[mention.end:mention.end + number.start()], re.I):
            # After a separator, a value written right before another quantity is that quantity's
            # ("1.5-1.6 eV band gap and 20-25% PCE": % is no band gap's unit), and one qualifying
            # another noun is that noun's ("~1.5 eV gap and 10-20 meV Urbach tail").
            rest = own[mention.end + hit[1]:]
            following = mentions[i + 1] if i + 1 < len(mentions) else None
            before_next = following is not None and not own[mention.end + hit[1]:following.start].strip(" -")
            if (before_next and not _unit_fits(hit[0].unit, mention.target) and _unit_fits(hit[0].unit, following.target)) \
                    or (not before_next and _qualified_noun(own, mention.end + hit[1], mention.end + hit[1])[1] == "after"
                        and not re.match(r"\s*[,.;?!)]", rest)):
                hit = None
        skip = 0
        if not hit:
            # "ZT at 600 K above 1": the measurement temperature first, the bound after it.
            t = re.match(rf"\s*(?:at|near|around|@)\s+{_NUM}\s*(?:K|°C|℃)(?![A-Za-z])", own[mention.end:limit], re.I)
            later = _window_after(own, replace(mention, end=mention.end + t.end()), limit) if t else None
            if later and not _is_temperature_unit(later[0].unit):
                hit, skip = (later[0], t.end() + later[1]), t.end()
        if not hit:
            floor = max([_sentence_of(bounds, mention.start)[0], mentions[i - 1].end if i else 0]
                        + [b for a, b in consumed if b <= mention.start])
            before = _window_before(own, mention, floor)
            if before:
                window, start = before
                conditions.append(replace(window, span=(start, mention.end)))
                consumed.append((start, mention.start))
            continue
        window, end = hit
        consumed_from = mention.end + skip
        if _is_temperature_unit(window.unit) and not mention.temperature_quantity:
            # "ZT near 870 K", "Seebeck below 427 °C": a measurement temperature, which the
            # quantity's own bound may follow ("ZT around 870 K above 1").
            later = _window_after(own, replace(mention, end=mention.end + end), limit)
            if later is None or _is_temperature_unit(later[0].unit):
                continue
            consumed_from = mention.end + end  # the temperature phrase stays for _temperatures
            window, end = later[0], end + later[1]
        if window.unit in ("°C", "℃"):
            window = replace(window, lo=window.lo and Bound(window.lo.value + 273.15, window.lo.inclusive),
                             hi=window.hi and Bound(window.hi.value + 273.15, window.hi.inclusive), unit="K")
        before_mention = own[max(0, mention.start - 25):mention.start].lower()
        after_mention = own[mention.end:mention.end + end].lower()
        # The quantity's own clause starts after the previous quantity's window phrase: "bandgap
        # values all between 1 and 2 eV and work function above 6.5" keeps 'all' on the band gap.
        clause_start = max(_sentence_of(bounds, mention.start)[0], mentions[i - 1].end if i else 0, phrase_end)
        quantifier, granularity = _quantifier(own, mention.start, mention.end, bounds, clause_start,
                                              mention.end + end)
        # The aggregate the window is on, named before the quantity ("maximum band gap below 2 eV",
        # "the minimum gap above 1 eV", "median"); after it only "peak" ("values peak at"). "at
        # most" and "maximum of" are comparators, not aggregates.
        aggregation = ("peak" if re.search(r"\b(?:peak|max(?:imum)?|highest|largest|biggest|greatest)\s+"
                                           r"(?:(?:recorded|reported|measured)\s+)?$",
                                           before_mention) or re.search(r"\bpeaks?\b", after_mention) or "최고" in before_mention else
                       "min" if re.search(r"\b(?:min(?:imum)?|lowest|smallest)\s+(?:(?:recorded|reported|measured)\s+)?$",
                                          before_mention) else
                       "median" if re.search(r"\bmedian\b|중앙값", before_mention + after_mention) else "point")
        if aggregation != "point":
            # "values peak at a value between 1 and 2 (the peak over every value)": the window is on
            # the aggregate, not on every value.
            quantifier, granularity = "any", "record"
        if window.quantifier == "every":
            quantifier = "every"  # "never exceeds 1 eV": no value above it
        if window.kind == "window" and _NEGATED_EXISTENCE_RE.search(own[clause_start:mention.start]):
            # "with no band gap above 1 eV", "none of whose values exceed 1 eV", "excluding those
            # with a band gap above 1 eV": every recorded value outside the stated window.
            flipped = _complement(window)
            if flipped is not None:
                window, quantifier = flipped, "every"
            else:
                window = Condition("approximate", target=window.target, lo=window.lo, unit=window.unit,
                                   qualitative="the complement of a two-sided window")
        for piece in (replace(window, extra=()),) + tuple(window.extra):
            conditions.append(replace(piece, span=(mention.start, mention.end + end), quantifier=quantifier,
                                      granularity=granularity, aggregation=aggregation))
        consumed.append((consumed_from, mention.end + end))
        position = mention.end + end
        while True:  # "above 1 eV but below 2 eV and above 1.7 eV": every continuation
            more = _continued_bound(own, mention, position, conditions[-1] if conditions else None)
            if more is None:
                break
            second, stop = more
            if _is_temperature_unit(second.unit) and not mention.temperature_quantity:
                break  # "ZT above 1 and below 700 K": the temperature, read below
            conditions.append(replace(second, span=(mention.start, stop),
                                      quantifier="every" if second.quantifier == "every" else quantifier,
                                      granularity=granularity, aggregation=aggregation))
            consumed.append((position, stop))
            position = stop
        phrase_end = position
    for pattern, label, lo, hi in _QUALITATIVE:
        for m in pattern.finditer(own):
            conditions.append(Condition("window", target=("bandgap",), lo=lo, hi=hi, unit="eV",
                                        qualitative=label, span=(m.start(), m.end())))
    for window in _unnamed_energy_windows(own, consumed, mentions, chain_windows, context):
        conditions.append(window)
        consumed.append(window.span)
    # In the order written: assignment reads each temperature against those before it.
    temperatures = sorted(_temperatures(own, consumed), key=lambda t: t.span)
    # A bare "300-400 K" is a measurement temperature only as an adjective of a temperature-bearing
    # quantity ("a 300-400 K ZT"); "a 30-40 K Tc" is the quantity's own value and "doped with
    # 0.5-1 K" is potassium.
    temperatures = [replace(t, qualitative="") for t in temperatures if t.qualitative != "bare range" or any(
        q.start >= t.span[1] and not own[t.span[1]:q.start].strip(" -") and set(q.target) & TEMPERATURE_BEARING
        for q in mentions)]
    ranks = _ranks(own, mentions, bounds)
    # One condition per owner: "at 600 K with ZT ... and Seebeck ..." bounds each of them.
    conditions += [replace(t, target=target) for t, owners in zip(
        temperatures, _temperature_assignments(own, temperatures, conditions, ranks, mentions, bounds))
                   for target in owners]
    conditions += ranks
    conditions += _exact_formulas(own, is_element)
    conditions += _substitution(own, is_element, parent_hint)
    conditions += _measured_only(own, bounds)
    segments = _count_segments(own)
    conditions += _device_conditions(own, segments)
    # "under 100 mW/cm2" is the light (read above), never a window on the quantity before it.
    conditions = [c for c in conditions if not (c.kind == "window" and re.fullmatch(
        r"mW/?cm(?:2|²|\^2)|suns?", (c.unit or "").replace(" ", ""), re.I))]
    # "PCE, Voc, Jsc and FF > 0" is the positive-J-V screen, not also a window "FF > 0".
    positive = [c.span for c in conditions if c.kind == "completeness" and c.completeness == "positive"]
    conditions = [c for c in conditions if not (c.kind == "window" and c.hi is None and c.lo is not None
                                                and c.lo.value == 0 and any(a <= c.span[0] < b for a, b in positive))]
    conditions = _without_screen_bounds(own, conditions)
    conditions = _implicit_rank_targets(conditions)
    # A ranking by a composition's median is the corpus-selection mode of screen_candidates;
    # no other tool ranks by medians.
    conditions = [replace(c, routed=True) if c.kind == "rank" and c.aggregation == "median" else c
                  for c in conditions]
    if _named_candidates(own, is_element):
        # "Of A, B and C, which has the widest band gap?": the population is the named candidates,
        # whose values the physics tools compute -- not a corpus selection (screen_candidates).
        conditions = [replace(c, routed=True) if c.kind in ("window", "rank") else c for c in conditions]
    if _CONCEPTUAL_RE.search(own) and not _CORPUS_WORD_RE.search(own):
        # "Why is it hard to push ZT far above 2?": a question about physics, not a population.
        conditions = []

    populations, subject_spans = _populations(own, is_element, sources_in)
    if len(populations) > 1:
        occurrences = _subject_occurrences(own, populations, sources_in)
        bound: List[Condition] = []
        for c in conditions:
            owners = _population_owners(own, bounds, occurrences, c)
            if owners is None or len(owners) >= len(populations):
                c.population = None
                bound.append(c)
            else:
                # A coordinated subset ("MP and OQMD ... against JARVIS ..."): one copy per owner.
                bound += [replace(c, population=i) for i in sorted(owners)]
        conditions = bound
    else:
        for c in conditions:
            c.population = 0

    def _owner(start, end) -> Optional[set]:
        if len(populations) <= 1:
            return {0}
        s_start, s_end = _sentence_of(bounds, start)
        inside = [(a, b, i) for a, b, i in occurrences if s_start <= a < s_end]
        owners = _adjective_owners(own, inside, (start, end))
        if owners is not None:
            return owners
        bound = _bound_population(own, bounds, occurrences, Condition("computed", span=(start, end)), len(populations))
        return None if bound is None else {bound}

    cancelled = _computed_contrasts(own, bounds, conditions, _owner)
    conditions = [c for c in conditions if not any(c is x for x in cancelled)]
    ranks = [c for c in conditions if c.kind == "rank"]
    conditions = [c for c in conditions if c.kind != "rank"]
    for population in sorted({c.population for c in ranks}, key=lambda x: (x is None, x or 0)):
        conditions += _fold_ranks([c for c in ranks if c.population == population])
    per_population = len(populations) > 1 and len(segments) == len(populations) and all(
        len({i for a, _, i in occurrences if seg[0] <= a < seg[1]}) == 1 for seg in segments)
    if per_population:
        # "How many in MP have band gap above 2 eV, and how many in OQMD?": the elliptical count
        # asks the same of its own source, so a condition stated in one count only is every count's.
        stated = [[c for c in conditions if a <= c.span[0] < b and c.kind != "rank"] for a, b in segments]
        elliptical = [not x and _elliptical(own[a:b], sources_in) for x, (a, b) in zip(stated, segments)]
        if sum(1 for x in stated if x) == 1 and all(e for x, e in zip(stated, elliptical) if not x):
            for c in next(x for x in stated if x):
                c.population = None
        # One count per population: conditions belong to populations, not to counts.
        segments = [(0, len(own))]
    conditions = _dedupe(conditions, segments if len(segments) > 1 else ())
    conditions = _intersected(conditions, segments if len(segments) > 1 else ())
    operations = _operations(own, conditions)
    if len(populations) > 1 and not per_population:
        for op in (o for o in operations if o.kind == "COUNT"):
            named = {i for a, _, i in occurrences if op.span[0] <= a < op.span[1]}
            owners = {conditions[i].population for i in op.conditions}
            # The count's own subject (not one it is "compared with"), else every population.
            op.population = (next(iter(named)) if len(named) == 1 and (not owners or owners & (named | {None}))
                             else None)
    counts = [o for o in operations if o.kind == "COUNT"]
    if len(counts) > 1:
        # A Korean "among ..." lead-in before the first count: what precedes it is every count's
        # population, not the first count's own condition.
        lead = re.search(r"\s중(?:에서)?\s|가운데\s", own[counts[0].span[0]:counts[0].span[1]])
        if lead:
            cut = counts[0].span[0] + lead.start()
            for index in [i for i in counts[0].conditions if conditions[i].span[1] <= cut + 1]:
                conditions[index].local, conditions[index].op = False, None
                counts[0].conditions.remove(index)
    if per_population:
        counts = [o for o in operations if o.kind == "COUNT"]
        for k, op in enumerate(counts):
            op.population, op.nested_in = k, None
        for c in conditions:
            c.local, c.op = False, None
    seen: List[str] = []
    for mention in mentions:
        for name in mention.target:
            if not mention.computed and not mention.device and name not in seen:
                seen.append(name)
    return QuestionPlan(own_text=own, pasted=pasted, populations=populations, conditions=conditions,
                        operations=operations, mentions=tuple(seen))


def _named_candidates(text: str, is_element) -> Tuple[str, ...]:
    """Return the candidates a question opens with ("Of A, B and C, ...", "Among A or B, ...")."""
    f = r"[A-Z][A-Za-z0-9.()]*"
    m = re.match(rf"\s*(?:of|among|between|from)\s+(?:the\s+\w+\s+)?({f}(?:\s*,\s*(?:and\s+|or\s+)?{f}|\s+(?:and|or)\s+{f})+)"
                 rf"\s*[,:]", text, re.I)
    if not m:
        return ()
    names = tuple(n for n in re.split(r"\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+", m.group(1)) if n)
    return names if len(names) >= 2 and all(_is_formula(n, is_element) for n in names) else ()


_CONCEPTUAL_RE = re.compile(r"^\s*(?:why\b|explain\b|how\s+(?:do|does|can|could|is|are)\b|what\s+(?:limits|"
                            r"makes|causes|determines)\b|왜\b)", re.I)
_CORPUS_WORD_RE = re.compile(r"\b(?:corpus|database|records?|recorded|collection|lamp|stored)\b|데이터베이스|레코드|"
                             r"\bdb\b", re.I)


def _without_screen_bounds(text: str, conditions: List[Condition]) -> List[Condition]:
    """Drop the PCE ceiling of a cell screen, which `CellScreen` reads whole.

    In "whose PCE is at most 34% and lies within a factor of two" the ceiling is part of the
    screen, not a second, separate window."""
    from materials.services.device_policy import CellScreen
    screen = CellScreen.from_text(text)
    if screen is None or screen.pce_max is None:
        return conditions
    return [c for c in conditions if not (c.kind == "window" and c.target == ("pce",) and c.lo is None
                                          and c.hi is not None and c.hi.value == screen.pce_max)]


def _tighter(a: Optional[Bound], b: Optional[Bound], lower: bool) -> Optional[Bound]:
    if a is None or b is None:
        return a or b
    if a.value != b.value:
        return (a if a.value > b.value else b) if lower else (a if a.value < b.value else b)
    return Bound(a.value, a.inclusive and b.inclusive)


def _intersected(conditions: List[Condition], segments: Sequence[Tuple[int, int]] = ()) -> List[Condition]:
    """Intersect windows on one quantity of one population ("above 1 eV and below 2 eV").

    Windows with the same unit and semantics are merged with both bounds held, unless they belong
    to separate COUNT operations ("how many above 1 eV, and how many below 0.5 eV")."""
    def segment(c):
        return next((i for i, (a, b) in enumerate(segments) if a <= c.span[0] < b), -1)

    out: List[Condition] = []
    for c in conditions:
        if c.kind == "window" and not c.local and not c.qualitative:
            for i, o in enumerate(out):
                if (o.kind == "window" and not o.local and not o.qualitative and o.target == c.target
                        and segment(o) == segment(c)
                        and o.population == c.population and o.unit == c.unit and o.quantifier == c.quantifier
                        and o.aggregation == c.aggregation and o.granularity == c.granularity):
                    out[i] = replace(o, lo=_tighter(o.lo, c.lo, True), hi=_tighter(o.hi, c.hi, False),
                                     span=(min(o.span[0], c.span[0]), max(o.span[1], c.span[1])))
                    break
            else:
                out.append(c)
        else:
            out.append(c)
    return out


def _implicit_rank_targets(conditions: List[Condition]) -> List[Condition]:
    """Give a ranking that names no quantity the one quantity the population is windowed on.

    "WF above 6.5 eV? Top five please" ranks by work function: highest first under a lower bound,
    lowest first under an upper bound. With no such window, or several, the ranking keeps no
    target and is not applied."""
    windows = [c for c in conditions if c.kind == "window" and not c.qualitative and c.target]
    if len(windows) != 1:
        return conditions
    w = windows[0]
    if (w.lo is None) == (w.hi is None):
        return conditions  # a two-sided window says nothing about which end is "top"
    out = []
    for c in conditions:
        if c.kind == "rank" and not c.target and not c.routed:
            direction = "desc" if w.lo is not None else "asc"
            c = replace(c, target=w.target, direction=c.direction or direction,
                        aggregation=w.aggregation if w.aggregation != "point" else c.aggregation)
        out.append(c)
    return out


def _dedupe(conditions: List[Condition], segments: Sequence[Tuple[int, int]] = ()) -> List[Condition]:
    """Keep identical conditions once.

    Identity includes the population, the COUNT operation (segment), unit and semantics, so two
    populations' equal windows both survive."""
    def segment(c):
        return next((i for i, (a, b) in enumerate(segments) if a <= c.span[0] < b), -1)

    def key(c):
        return (c.kind, c.target, c.lo, c.hi, c.unit, c.formulas, c.completeness, c.irradiance_mw_cm2,
                c.quantifier, c.aggregation, c.granularity, c.qualitative, c.population, segment(c))

    out: List[Condition] = []
    for c in conditions:
        if any(key(o) == key(c) for o in out):
            continue
        out.append(c)
    return out


def _combine(earlier: List[Condition], new: List[Condition], own_text: str,
             names_of: Optional[Callable[[str], set]] = None) -> List[Condition]:
    """A follow-up's conditions over the population it refers to.

    The same kind and target replaces, a new kind adds, a substitution replaces exact-composition
    eligibility (the exact formula becomes its parent), and an explicit release removes."""
    if _RELEASE_TEMPERATURE_RE.search(own_text):
        earlier = [c for c in earlier if c.kind != "temperature"]
    if _releases_measured(own_text):
        earlier = [c for c in earlier if c.kind != "measured_only"]
        new = [c for c in new if c.kind != "measured_only"]
    for m in list(_RELEASE_PROPERTY_RE.finditer(own_text)) + list(_RELEASE_PROPERTY_KO_RE.finditer(own_text)):
        # The released quantity read with the same vocabulary as conditions ("band gap" -> bandgap).
        released = names_of(m.group("what")) if names_of else {m.group("what").strip().replace(" ", "_")}
        earlier = [c for c in earlier if not (c.kind == "window" and set(c.target) & released)]
    split = {}
    for o in earlier:
        if o.kind == "temperature" and o.target and o.target != ("device",):
            split.setdefault((o.span, o.lo, o.hi), []).append(o)
    statement = next((v for v in split.values() if len(v) > 1), None) if len(split) == 1 else None
    if statement and not any(n.kind == "temperature" for n in new):
        # "... ZT ... and Seebeck ... at 600 K" then "now also thermal conductivity below 1": one
        # temperature stated for every quantity stays every quantity's, the added one's too.
        new = list(new) + [replace(statement[0], target=n.target) for n in new if n.kind in ("window", "rank")
                           and n.target and set(n.target) & TEMPERATURE_BEARING]  # rankings too
    owned = {x for n in new if n.kind == "temperature" and n.target for x in n.target}
    shared = [o for o in earlier if o.kind == "temperature" and not o.target]
    if owned and shared:
        # "Now ZT at 700 K" after "... ZT ... and Seebeck ... at 600 K": ZT's is the new one and
        # Seebeck keeps 600 K; the shared temperature stays with each other temperature-bearing
        # quantity.
        others = dict.fromkeys(c.target for c in list(earlier) + list(new) if c.kind in ("window", "rank") and c.target
                               and set(c.target) & TEMPERATURE_BEARING and not set(c.target) & owned)
        earlier = [o for o in earlier if not (o.kind == "temperature" and not o.target)] + [
            replace(o, target=target) for o in shared for target in others]
    kept = []
    for old in earlier:
        replaced = any(n.kind == old.kind and (n.population is None or n.population == old.population)
                       and (n.target == old.target or n.kind in (
                           "rank", "exact_formula", "substitution", "completeness", "illumination", "measured_only")
                            # A temperature replaces the one of the same quantity (one
                            # stated for every quantity replaces them all), never another's.
                            or (n.kind == "temperature" and not n.target)) for n in new)
        if old.kind == "exact_formula" and any(n.kind == "substitution" for n in new):
            replaced = True
        if old.kind == "rank" and any(n.kind == "rank" for n in new):
            replaced = True
        if not replaced:
            kept.append(old)
    merged_new = []
    old_rank = next((c for c in earlier if c.kind == "rank"), None)
    for n in new:
        if n.kind == "rank" and old_rank is not None:
            n = replace(n, target=n.target or old_rank.target, n=n.n or old_rank.n,
                        aggregation=n.aggregation if n.target else old_rank.aggregation)
        merged_new.append(n)
    return kept + merged_new


def parse_question(text: str, *, property_patterns, is_element: Callable[[str], bool],
                   sources_in: Callable[[str], List[str]], history: Sequence[str] = (),
                   cation_tokens: Sequence[str] = ()) -> QuestionPlan:
    """Return the QuestionPlan of `text`, its population resolved along `history`.

    `history` holds earlier user messages, oldest first. A turn that refers back ("them" with
    nothing of its own before it, "redo that", "now ...") continues the population of the turn
    before; a turn that names its own subject starts a new one; a release ("drop the source
    filter", "any temperature") removes what it names."""
    global _CATION_TOKENS
    _CATION_TOKENS = tuple(cation_tokens)

    def names_of(phrase: str) -> set:
        return {name for m in _mentions(phrase, property_patterns) for name in m.target}

    chain: List[Condition] = []
    chain_pops: List[Population] = []
    seen: List[str] = []
    for earlier in history:
        p = _read(earlier, property_patterns=property_patterns, is_element=is_element, sources_in=sources_in,
                  parent_hint=_parent_of(chain), chain_windows=tuple(c.target for c in chain if c.kind == "window"),
                  context=" ".join(seen))
        seen.append(earlier)
        chain, chain_pops, _, _ = _advance(chain, chain_pops, p, sources_in, names_of)
    current = _read(text, property_patterns=property_patterns, is_element=is_element, sources_in=sources_in,
                    parent_hint=_parent_of(chain), chain_windows=tuple(c.target for c in chain if c.kind == "window"),
                    context=" ".join(seen))
    chain_after, pops_after, continued, ambiguous = _advance(chain, chain_pops, current, sources_in, names_of)
    if ambiguous:
        current.unresolved = (ambiguous,)
    elif continued:
        if len(pops_after) == 1:
            for c in chain_after:
                c.population = 0 if c.population is not None else None
        current.conditions = chain_after
        current.populations = pops_after
        current.operations = _operations(current.own_text, current.conditions)
        if len([o for o in current.operations if o.kind == "COUNT"]) < 2:
            # "Now the same for OQMD" after two separate counts: the counts carry over, each with
            # its own conditions, rather than leaving local conditions that no operation owns.
            groups: Dict[int, List[int]] = {}
            parents: Dict[int, Optional[int]] = {}
            for index, c in enumerate(current.conditions):
                if c.local and c.op is not None:
                    groups.setdefault(c.op, []).append(index)
                    parents[c.op] = c.op_parent
            if groups:
                order = sorted(groups)
                current.operations = [o for o in current.operations if o.kind != "COUNT"] + [
                    # ... nested as they were ("how many of those"), not flattened
                    Operation("COUNT", population=None if len(pops_after) > 1 else 0, conditions=groups[k],
                              nested_in=order.index(parents[k]) if parents.get(k) in groups else None)
                    for k in order]
    else:
        anaphor = _ANAPHORA_RE.search(current.own_text)
        if anaphor and not current.pasted and not _own_antecedent(current, anaphor, sources_in) and not (
                current.populations[0].sources or current.populations[0].formulas):
            current.unresolved = (anaphor.group(0).lower(),)
    return current


_BARE_BOUND_RE = re.compile(
    rf"(?P<w>{_LO_INCL}|{_HI_INCL}|{_LO_STRICT}|{_HI_STRICT})\s*(?P<a>{_NUM})\s*(?P<unit>{_UNIT})?"
    rf"|(?P<a2>{_NUM})\s*(?P<unit2>{_UNIT})?\s*(?P<post>이하|미만|이상|초과|or\s+(?:more|higher|above|greater|less|lower|below)|"
    rf"and\s+(?:above|below))"
    rf"|(?P<a3>{_NUM})\s*(?P<unit3>{_KNOWN_UNIT})\s*(?:instead|대신)", re.I)


def _bare_bounds(p: QuestionPlan, chain: List[Condition]) -> Tuple[List[Condition], str]:
    """Read a follow-up's bound that names no quantity ("only those above 3 eV", "1.2 eV instead").

    The bound applies to the one window quantity the conversation has, replacing that side and
    keeping the other. ([], "") when there is none; ([], reason) when the quantity is ambiguous."""
    if any(c.kind in ("window", "approximate") for c in p.conditions):
        return [], ""
    taken = [c.span for c in p.conditions]
    found = []
    for m in _BARE_BOUND_RE.finditer(p.own_text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        unit = m["unit"] or m["unit2"] or m["unit3"] or ""
        if _is_temperature_unit(unit):
            continue  # a temperature is read as one already
        if _COUNTED_RE.match(p.own_text, m.end()) or (
                unit and unit.lower() not in _NOT_A_UNIT and not re.fullmatch(_KNOWN_UNIT, unit, re.I)
                and not any(c.kind == "window" and _unit_word(c.unit).lower() == unit.lower() for c in chain)):
            continue  # "at least 10 of them", "more than 3 elements": a count, not a bound
        if re.search(r"\b(?:top|first|best)\s*$", p.own_text[:m.start()], re.I):
            continue
        found.append((m, _unit_word(unit) if unit.lower() not in _NOT_A_UNIT else ""))
    if not found:
        return [], ""
    windows = [c for c in chain if c.kind == "window"]
    out = []
    for m, unit in found:
        targets = list(dict.fromkeys(c.target for c in windows))
        if len(targets) > 1 and unit:
            targets = list(dict.fromkeys(c.target for c in windows if _unit_word(c.unit) == unit)) or targets
        if len(targets) != 1:
            return [], "a bound with no quantity named" if targets else ""
        for old in [c for c in windows if c.target == targets[0]]:
            value = _num(m["a"] or m["a2"] or m["a3"])
            if m["w"]:
                side, inclusive, _ = _comparator(m["w"], p.own_text[:m.start("w")])
            elif m["post"]:
                post = m["post"].lower()
                side = "hi" if re.search(r"이하|미만|less|lower|below", post) else "lo"
                inclusive = not re.search(r"미만|초과", post)
            else:  # "1.2 eV instead": the side the earlier one-sided window had
                if (old.lo is None) == (old.hi is None):
                    return [], "a bound with no side stated against a two-sided window"
                side, inclusive = ("lo", old.lo.inclusive) if old.lo else ("hi", old.hi.inclusive)
            bound = Bound(value, inclusive)
            if unit and old.unit and _unit_word(unit) != _unit_word(old.unit) and (old.hi if side == "lo" else old.lo):
                # "now below 2000 meV" after "above 1 eV": each side keeps its own unit, intersected
                # once converted.
                out.append(replace(old, lo=None if side == "lo" else old.lo, hi=None if side == "hi" else old.hi,
                                   span=(m.start(), m.end()), qualitative=""))
                out.append(replace(old, lo=bound if side == "lo" else None, hi=bound if side == "hi" else None,
                                   unit=unit, span=(m.start(), m.end()), qualitative=""))
                continue
            out.append(replace(old, lo=bound if side == "lo" else old.lo, hi=bound if side == "hi" else old.hi,
                               unit=unit or old.unit, span=(m.start(), m.end()), qualitative=""))
    return out, ""


def _own_antecedent(p: QuestionPlan, anaphor, sources_in) -> bool:
    """Whether a "them" / "those" has something in its own turn to point back to: a stated condition
    (not the ranking that contains it), a formula, a source or a population noun before it."""
    before = p.own_text[:anaphor.start()]
    return (any(c.span[0] < anaphor.start() for c in p.conditions if c.kind != "rank")
            or bool(_FORMULA_RE.search(before)) or bool(sources_in(before))
            or bool(re.search(r"\b(?:materials?|compounds?|records?|samples?|emitters?|devices?|cells?|mofs?|"
                              r"monolayers?|perovskites?|thermoelectrics?)\b|물질|소자|샘플|레코드", before, re.I)))


def _advance(chain, chain_pops, p: QuestionPlan, sources_in, names_of=None):
    """Advance the conversation by one turn.

    Returns (conditions, populations, whether this turn continued the previous population(s),
    ambiguity). A turn naming its own subject with its own conditions is a new request even after
    "now"; "redo that for OQMD" (a new subject, nothing else) keeps the conditions and swaps the
    subject; a follow-up to a comparison keeps every compared population."""
    anaphor = _ANAPHORA_RE.search(p.own_text)
    own = p.populations[0]
    chain_population = chain_pops[0] if chain_pops else None
    known = [(cp.sources, cp.formulas) for cp in chain_pops]
    new_subject = bool(own.sources or own.formulas) and (not chain_pops or (own.sources, own.formulas) not in known)
    # "the same for OQMD", "redo", "again"; not "the same pore size distribution".
    explicit_same = bool(re.search(
        r"\b(?:the\s+)?same\s+(?:for|but|thing|again|query|question|request|search|list|count|analysis|with|in|on|as|"
        r"conditions?|criteria|filters?)\b|\bdo\s+the\s+same\b|\b(?:the\s+)?same\s*[?.!]?\s*$|\bredo\b|\bagain\b|"
        r"다시|똑같|마찬가지|같은\s*조건", p.own_text, re.I))
    new_family = (own.noun in _DISTINCT_FAMILIES and chain_population is not None
                  and own.noun not in {cp.noun for cp in chain_pops})
    if new_family:
        new_subject = True  # "now list perovskites ..." after monolayers: another family
    # "those" after a pasted list points at the pasted rows, not at the previous turn.
    dangling = bool(anaphor and not p.pasted and not _own_antecedent(p, anaphor, sources_in))
    # "what about above 1.2 eV?", "1.2 eV instead", "include DFT values too": a follow-up with no
    # other subject or quantity of its own.
    bare_follow_up = (not p.conditions and not (own.sources or own.formulas) and bool(chain)
                      and bool(_BARE_BOUND_RE.search(p.own_text) or _releases_measured(p.own_text))
                      and _only_a_follow_up(p.own_text, sources_in))
    what_about = bool(re.search(r"^\s*(?:what|how)\s+about\b|\binstead\b", p.own_text, re.I)
                      and _only_a_follow_up(p.own_text, sources_in))
    continuation = bool(_CONTINUATION_RE.search(p.own_text) or explicit_same or bare_follow_up or what_about)
    refers_back = dangling or (continuation and (explicit_same or not new_family)
                               and not (new_subject and p.conditions and not explicit_same))
    if refers_back and (chain or chain_pops):
        bare, problem = _bare_bounds(p, chain)
        if problem:
            return chain, chain_pops, False, problem
        new = list(p.conditions) + bare
        if new_subject:
            if len(chain_pops) > 1 and any(c.population is not None for c in chain if c.kind != "rank"):
                # "Same for JARVIS" after "MP above 2 eV against OQMD below 1 eV" is ambiguous,
                # unless the compared populations stated one set between them ("MP and OQMD above
                # 2 eV against JARVIS").
                def sig(c):
                    # Everything a condition says (light, completeness, formulas, substitution), but
                    # not where it was written or which population/count it was numbered under.
                    return repr(replace(c, span=(0, 0), population=None, op=None, op_parent=None))
                # A compared population that stated no condition is a set of its own: "MP above
                # 2 eV against OQMD" then "same for JARVIS" (like MP or like OQMD?) is asked, never
                # guessed.
                sets = {i: set() for i in range(len(chain_pops))}
                for c in chain:
                    if c.kind != "rank" and c.population is not None:
                        sets.setdefault(c.population, set()).add(sig(c))
                distinct = {frozenset(v) for v in sets.values()}
                if len(distinct) != 1:
                    return chain, chain_pops, False, "same (which compared population)"
                keep, seen = [], set()
                for c in chain:
                    if c.population is None or (c.kind != "rank" and sig(c) not in seen):
                        if c.population is not None:
                            seen.add(sig(c))
                        keep.append(replace(c, population=None))
                chain = keep
            base = chain_population or own
            population = replace(base, sources=own.sources or base.sources, formulas=own.formulas or base.formulas,
                                 noun=own.noun or base.noun)
            if _RELEASE_SOURCE_RE.search(p.own_text):
                population = replace(population, sources=())
            return _combine(chain, new, p.own_text, names_of), [population], True, ""
        pops = list(chain_pops) or [own]
        if len(pops) == 1:
            pops = [replace(pops[0], noun=own.noun or pops[0].noun)]
            if _RELEASE_SOURCE_RE.search(p.own_text):
                pops = [replace(pops[0], sources=())]
        else:
            # The follow-up's own conditions: of the compared population it names, else of all.
            mine = [i for i, cp in enumerate(pops) if (own.sources or own.formulas)
                    and (cp.sources, cp.formulas) == (own.sources, own.formulas)]
            for c in p.conditions:
                c.population = mine[0] if mine else None
        return _combine(chain, new, p.own_text, names_of), pops, True, ""
    if new_subject or p.conditions or own.noun in _FAMILY_NOUNS:
        return list(p.conditions), list(p.populations), False, ""
    return chain, chain_pops, False, ""


def _only_a_follow_up(text: str, sources_in) -> bool:
    """Whether a turn says nothing but a bound, a release or a new subject.

    "what about above 1.2 eV?", "1.2 eV instead", "what about OQMD?" and "include DFT values too"
    qualify; "what about the stability of CsPbI3?" and "why does efficiency drop under 85 %
    humidity?" open a topic of their own."""
    rest = _BARE_BOUND_RE.sub(" ", text)
    rest = _RELEASE_MEASURED_RE.sub(" ", rest)
    rest = _FORMULA_RE.sub(" ", rest)
    for source in sources_in(text):
        rest = re.sub(re.escape(source), " ", rest, flags=re.I)
    rest = re.sub(r"\b(?:what|how|about|and|or|then|instead|only|just|those|these|them|ones|please|now|with|the|values?|"
                  r"too|also|as\s+well|in|for|of|show|list|give|me|again|fine|ok|okay|are|is|data)\b|"
                  r"그럼|이번엔|대신|만|은|는|도|으로|로|에서|[,?.!:]", " ", rest, flags=re.I)
    return not re.search(r"[A-Za-z가-힣]{2,}", rest)


def _parent_of(chain: List[Condition]) -> Optional[str]:
    for c in reversed(chain):
        if c.kind == "exact_formula" and c.formulas:
            return c.formulas[0]
        if c.kind == "substitution" and c.substitution and c.substitution.get("parent"):
            return c.substitution["parent"]
    return None
